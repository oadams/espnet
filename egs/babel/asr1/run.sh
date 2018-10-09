#!/bin/bash

# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
seed=1
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

# network archtecture
# encoder related
etype=vggblstmp # encoder architecture type
elayers=4
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers

tdnn_offsets="0 -1,0,1 -1,0,1 -3,0,3 -3,0,3 -3,0,3 -3,0,3"
tdnn_odims="625 625 625 625 625 625 625"
tdnn_prefinal_affine_dim=625
tdnn_final_affine_dim=3000
kaldi_mdl='text_8k.mdl'

# decoder related
dlayers=1
dunits=300

# attention related
atype=location
adim=320
awin=5
aheads=4
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.5

# label smoothing
lsm_type=unigram
lsm_weight=0.05

# minibatch related
batchsize=30
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=20

# rnnlm related
use_lm=false
lm_layers=2
lm_units=650
lm_opt=sgd        # or adam
lm_batchsize=256  # batch size in LM training
lm_epochs=20      # if the data size is large, we can reduce this
lm_maxlen=100     # if sentence length > lm_maxlen, lm_batchsize is automatically reduced
lm_resume=        # specify a snapshot file to resume LM training
lmtag=            # tag for managing LMs

# decoding parameter
lm_weight=1.0
beam_size=20
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# exp tag
tag="" # tag for managing experiments.

langs="101 102 103 104 105 106 202 203 204 205 206 207 301 302 303 304 305 306 401 402 403"
recog="107 201 307 404"
upsample=true
extractor=extractor
use_ivectors=false

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Train Directories
train_set=train
train_dev=dev

tdnn_odims_array=( ${tdnn_odims} )
tdnn_offsets_array=( ${tdnn_offsets} )

if [ ${#tdnn_odims_array[@]} -ne ${#tdnn_offsets_array[@]} ]; then
  echo "tdnn_odims_array and tdnn_offsets_array must have the same number of elements"
  exit 1
fi


# LM Directories
if [ -z ${lmtag} ]; then
    lmtag=${lm_layers}layer_unit${lm_units}_${lm_opt}_bs${lm_batchsize}
fi
lmexpdir=exp/train_rnnlm_${backend}_${lmtag}
lm_train_set=data/local/train.txt
lm_valid_set=data/local/dev.txt

recog_set=""
for l in ${recog}; do
  recog_set="eval_${l} ${recog_set}"
done
recog_set=${recog_set%% }

if [ $stage -le 0 ]; then
  echo "stage 0: Setting up individual languages"
  ./local/setup_languages.sh --langs "${langs}" --recog "${recog}" --FLP true
  if $upsample; then
      for x in ${train_set} ${train_dev} ${recog_set}; do
        sed -i.bak -e "s/$/ sox -R -t wav - -t wav - rate 16000 dither | /" data/${x}/wav.scp
      done
  fi
fi


feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ $stage -le 1 ]; then
  echo "stage 1: Feature extraction"
  mfccdir=mfcc_pitch_online
  # Generate the fbank features
  for x in ${train_set} ${train_dev} ${recog_set}; do
      steps/make_mfcc_pitch_online.sh --cmd "$train_cmd" \
                                      --mfcc-config conf/mfcc_hires.conf --nj 40 \
                                      data/${x} exp/make_mfcc_pitch_online/${x} ${mfccdir}
      
      steps/compute_cmvn_stats.sh data/${x}
      utils/fix_data_dir.sh data/${x}

      if $use_ivectors; then
          utils/data/limit_feature_dim.sh 0:39 data/${x} data/${x}_nopitch
          steps/compute_cmvn_stats.sh data/${x}_nopitch exp/make_mfcc/${x}_nopitch ${mfccdir}
      
          utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
              data/${x}_nopitch data/${x}_nopitch_max2

          steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 20 --repeat true \
              data/${x}_nopitch_max2 ${extractor} data/${x}_ivectors
      fi
  done

  # compute global CMVN
  compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark
  ./utils/fix_data_dir.sh data/${train_set}

  exp_name=`basename $PWD`
  # dump features for training
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
  utils/create_split_dir.pl \
      /export/b{10,11,12,13}/${USER}/espnet-data/egs/babel/${exp_name}/dump/${train_set}/delta${do_delta}/storage \
      ${feat_tr_dir}/storage
  fi
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
  utils/create_split_dir.pl \
      /export/b{10,11,12,13}/${USER}/espnet-data/egs/babel/${exp_name}/dump/${train_dev}/delta${do_delta}/storage \
      ${feat_dt_dir}/storage
  fi
  
  dump.sh --cmd "$train_cmd" --nj 20 --do_delta $do_delta --ivectors data/${train_set}_ivectors/ivector_online.scp \
      data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
  dump.sh --cmd "$train_cmd" --nj 10 --do_delta $do_delta --ivectors data/${train_dev}_ivectors/ivector_online.scp \
      data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
  for rtask in ${recog_set}; do
      feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
      dump.sh --cmd "$train_cmd" --nj 10 --do_delta $do_delta --ivectors data/${rtask}_ivectors/ivector_online.scp \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
  done
fi

dict=data/lang_1char/${train_set}_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    cut -f 2- data/${train_set}/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | grep -v '<unk>' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    ivector_opts=
    if $use_ivectors; then
      ivector_opts="--ivectors ${feat_tr_dir}/ivectors_online.scp"
    fi

    echo "make json files"
    mkjson.py --non-lang-syms ${nlsyms} ${ivector_opts} \
              ${feat_tr_dir}/feats.scp data/${train_set} ${dict} \
              > ${feat_tr_dir}/data.json

    mkjson.py --non-lang-syms ${nlsyms} ${ivector_opts} \
              ${feat_dt_dir}/feats.scp data/${train_dev} ${dict} \
              > ${feat_dt_dir}/data.json

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        mkjson.py --non-lang-syms ${nlsyms} ${ivector_opts} \
              ${feat_recog_dir}/feats.scp data/${rtask} ${dict} \
              > ${feat_recog_dir}/data.json
    done
fi

if $use_lm; then
  lm_train_set=data/local/train.txt
  lm_valid_set=data/local/dev.txt

  # Make train and valid
  text2token.py --nchar 1 \
                --space "<space>" \
                --non-lang-syms data/lang_1char/non_lang_syms.txt \
                <(cut -d' ' -f2- data/${train_set}/text | head -100) \
                > ${lm_train_set}

  text2token.py --nchar 1 \
                --space "<space>" \
                --non-lang-syms data/lang_1char/non_lang_syms.txt \
                <(cut -d' ' -f2- data/${train_dev}/text | head -100) \
                > ${lm_valid_set}

  if [ ${ngpu} -gt 1 ]; then
        echo "LM training does not support multi-gpu. signle gpu will be used."
  fi

  ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
          lm_train.py \
          --ngpu ${ngpu} \
          --backend ${backend} \
          --verbose 1 \
          --outdir ${lmexpdir} \
          --train-label ${lm_train_set} \
          --valid-label ${lm_valid_set} \
          --resume ${lm_resume} \
          --layer ${lm_layers} \
          --unit ${lm_units} \
          --opt ${lm_opt} \
          --batchsize ${lm_batchsize} \
          --epoch ${lm_epochs} \
          --maxlen ${lm_maxlen} \
          --dict ${dict}
fi


if [ -z ${tag} ]; then
    expdir=exp/${train_set}_${backend}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
else
    expdir=exp/${train_set}_${backend}_${tag}
fi
mkdir -p ${expdir}

if [ ${stage} -le 3 ]; then
    echo "stage 3: Network Training"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --seed ${seed} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --tdnn-offsets "${tdnn_offsets}" \
        --tdnn-odims "${tdnn_odims}" \
        --tdnn-prefinal-affine-dim ${tdnn_prefinal_affine_dim} \
        --tdnn-final-affine-dim ${tdnn_final_affine_dim} \
        --kaldi-mdl ${kaldi_mdl} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --adim ${adim} \
        --awin ${awin} \
        --aheads ${aheads} \
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --mtlalpha ${mtlalpha} \
        --lsm-type ${lsm_type} \
        --lsm-weight ${lsm_weight} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --opt ${opt} \
        --epochs ${epochs}
fi


if [ ${stage} -le 4 ]; then
    echo "stage 4: Decoding"
    nj=64

    extra_opts=""
    if $use_lm; then
      extra_opts="--rnnlm ${lmexpdir}/rnnlm.model.best --lm-weight ${lm_weight} ${extra_opts}"
    fi

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}
        if $use_lm; then
            decode_dir=${decode_dir}_rnnlm${lm_weight}_${lmtag}
        fi
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --ctc-weight ${ctc_weight} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            ${extra_opts} &
        wait

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi

