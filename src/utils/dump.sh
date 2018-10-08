#!/bin/bash

# Copyright 2017 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

cmd=run.pl
do_delta=false
nj=1
verbose=0
compress=true
write_utt2num_frames=true
ivectors=

. utils/parse_options.sh

scp=$1
cmvnark=$2
logdir=$3
dumpdir=$4

if [ $# != 4 ]; then
    echo "Usage: $0 <scp> <cmvnark> <logdir> <dumpdir>"
    exit 1;
fi

mkdir -p $logdir
mkdir -p $dumpdir

dumpdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' ${dumpdir} ${PWD}`

for n in $(seq $nj); do
    # the next command does nothing unless $dumpdir/storage/ exists, see
    # utils/create_data_link.pl for more info.
    utils/create_data_link.pl ${dumpdir}/feats.${n}.ark
    if [ ! -z $ivectors ] && [ -f $ivectors ]; then
        ./utils/create_data_link.pl ${dumpdir}/ivectors_online.${i}.ark
    fi
done

if $write_utt2num_frames; then
    write_num_frames_opt="--write-num-frames=ark,t:$dumpdir/utt2num_frames.JOB"
else
    write_num_frames_opt=
fi

# split scp file
split_scps=""
split_scps_ivector=""
for n in $(seq $nj); do
    split_scps="$split_scps $logdir/feats.$n.scp"
    split_scps_ivector="$split_scps_ivector $logdir/ivectors_online.$n.scp"
done

utils/split_scp.pl $scp $split_scps || exit 1;
if [ ! -z $ivectors ] && [ -f $ivectors ]; then
    utils/split_scp.pl $ivectors $split_scps_ivector || exit 1;
fi

# Feature extraction
feat_cmd="copy-feats scp:${logdir}/feats.JOB.scp ark:- |"

# Only add cmvn is no ivectors are provided
if [ -z $ivectors ] && [ -f $ivectors ]; then
    echo "HERE"
    feat_cmd="${feat_cmd} apply-cmvn --norm-vars=true $cmvnark ark:- ark:- |"
fi

# Add deltas
if ${do_delta}; then
    feat_cmd="${feat_cmd} add-deltas ark:- ark:- |"
fi

# Store output
feat_cmd="${feat_cmd} copy-feats --compress=${compress} --compression-method=2 ${write_num_frames_opt} ark:- ark,scp:${dumpdir}/feats.JOB.ark,${dumpdir}/feats.JOB.scp"

# Extract features
$cmd JOB=1:${nj} ${logdir}/dump_feature.JOB.log \
  ${feat_cmd} || exit 1

if [ ! -z $ivectors ] && [ -f $ivectors ]; then
    $cmd JOB=1:$nj $logdir/dump_ivectors.JOB.log \
        copy-feats --compress=$compress --compression-method=2 \
            scp:$logdir/ivectors_online.JOB.scp ark,scp:${dumpdir}/ivectors_online.JOB.ark,${dumpdir}/ivectors_online.JOB.scp \
        || exit 1

    cat ${dumpdir}/ivectors_online.*.scp > ${dumpdir}/ivectors_online.scp
fi

# concatenate scp files
cat ${dumpdir}/feats.*.scp > ${dumpdir}/feats.scp

if $write_utt2num_frames; then
    for n in $(seq $nj); do
        cat $dumpdir/utt2num_frames.$n || exit 1;
    done > $dumpdir/utt2num_frames || exit 1
    rm $dumpdir/utt2num_frames.* 2>/dev/null
fi

# remove temp scps
rm $logdir/feats.*.scp 2>/dev/null
if [ ${verbose} -eq 1 ]; then
    echo "Succeeded dumping features for training"
fi
