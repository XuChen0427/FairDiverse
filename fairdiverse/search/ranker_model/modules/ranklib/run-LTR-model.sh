#!/bin/bash
#################################################
#						#
# Properties for complete routine		#
#						#
#################################################
#################################################
#						#
# Functions for complete routine		#
#						#
#################################################
function writeInfoFile {
	# $1 is ranker name
	# $2...n are hyper parameters for training
	
	infofile="$dataRootPath/$1/training-parameters-info.txt"
	case "$1" in
		#writing infofile to evaluate quality of training parameters
		"MART" | "LambdaMART" )
		if [ -e $infofile ]; then
  		  echo "$2; $3; $4;" >> $infofile
		else
		  echo "MART PARAMETERS WITH MAP EVALUATION" > $infofile
		  echo "Shrinkage; Trees; Leafs; MAP; NDCG" >> $infofile
  	  	  echo "$2; $3; $4;" >> $infofile
		fi;;
		"ListNet" )
		if [ -e $infofile ]; then
                  echo "$2; $3;" >> $infofile
		else
		  echo "LISTNET PARAMETERS WITH MAP EVALUATION" > $infofile
		  echo "Learning Rate; Epochs; MAP; NDCG" >> $infofile
  	  	  echo "$2; $3;" >> $infofile
		fi;;
		"CoordinateAscent" )
		if [ -e $infofile ]; then
                  echo "$2; $3; $4;" >> $infofile
		else
		  echo "COORDINATE ASCENT PARAMETERS WITH MAP EVALUATION" > $infofile
		  echo "Random Restarts; Iteration; Performance Tolerance; MAP; NDCG" >> $infofile
  	  	  echo "$2; $3; $4;" >> $infofile
		fi;;
	esac
}
function createDirs {
	mkdir $1
	mkdir $1/models
	mkdir $1/predictions
}

function runRanker {
	# first argument experiment directory, 
	# second argument ranker name, 
	# third ranker number, 
	# fourth timestamp in training data files
	# last n arguments are hyper parameters for individual rankers
	echo " "
	echo "########################################"
	echo " "
	echo "Start training with ranker $2 using file $4"
	echo " "
	echo "########################################"

  echo $3
  echo $2
  echo $5
  echo $6
  echo $1
  echo $4
  echo $RELEVANCE_LABEL
  echo $RANK_LIB_FILE
  echo $EVAL_M
  echo $dataRootPath

	case "$2" in
		"MART" | "LambdaMART" )
		echo "LambdaMART";
		#if [ "$#" -ne 7 ]; then echo "ERROR: incorrect number of arguments for MART or LambdaMART"; exit -1; fi
		java -jar $RANK_LIB_FILE -train $dataRootPath/train_ranklib.txt \
			-ranker 6 -shrinkage 0.1 -tree 1000 -leaf 10 -gmax $RELEVANCE_LABEL \
			-metric2t $EVAL_M -metric2T $EVAL_M -save $1/models/model.txt;;
		"ListNet" )
		if [ "$#" -ne 10 ]; then echo "ERROR: incorrect number of arguments for ListNet"; exit -1; fi

		java -jar $RANK_LIB_FILE -train $dataRootPath/train_ranklib.txt \
			-ranker $3 -lr $5 -epoch $6 -gmax $RELEVANCE_LABEL \
		  -metric2T $EVAL_M -save $1/models/$4_model.txt;;
#
    "AdaRank")
    if [ "$#" -ne 6 ]; then echo "ERROR: incorrect number of arguments for ListNet"; exit -1; fi

    java -jar $RANK_LIB_FILE -train $dataRootPath/train_ranklib.txt \
			-ranker $3 -lr $5 -round $6 -gmax $RELEVANCE_LABEL \
			-metric2t $EVAL_M -metric2T $EVAL_M -save $1/models/model.txt;;

		"CoordinateAscent" )
		if [ "$#" -ne 7 ]; then echo "ERROR: incorrect number of arguments for CoordinateAscent"; exit -1; fi
		java -jar $RANK_LIB_FILE -train $dataRootPath/train_ranklib.txt \
			-ranker $3 -r $5 -i $6 -tolerance $7 -gmax $RELEVANCE_LABEL \
			-metric2t $EVAL_M -metric2T $EVAL_M -save $1/models/$model.txt;;
	"RankNet" )
	echo "RankNet"
	java -jar $RANK_LIB_FILE -train $dataRootPath/train_ranklib.txt -ranker $3 -epoch $6 -layer 2 -lr $5 -node 10 -gmax $RELEVANCE_LABEL \
			-metric2t $EVAL_M -metric2T $EVAL_M -save $1/models/model.txt;;
		* )
		# anything else
		# if [ "$#" -ne 4 ]; then exit -1; fi
		java -jar $RANK_LIB_FILE -train $dataRootPath/train_ranklib.txt -ranker $3 -gmax $RELEVANCE_LABEL \
			-metric2t $EVAL_M -metric2T $EVAL_M -save $1/models/model.txt;;
	esac
	
	echo " "	
	echo "########################################"
	echo " "
	echo "Predicting scores for ranker $2 using file $4"
	echo " "
	echo "########################################"
	java -jar $RANK_LIB_FILE -load $1/models/model.txt -rank $dataRootPath/test_ranklib.txt \
		-indri $1/predictions/prediction.txt

	echo " "
	echo "########################################"
	echo " "
	echo "Predicting scores for ranker $2 using file $4"
	echo " "
	echo "########################################"
	java -jar $RANK_LIB_FILE -load $1/models/model.txt -rank $dataRootPath/train_ranklib.txt \
		-indri $1/predictions/prediction_train.txt

  echo " "
	echo "########################################"
	echo " "
	echo "Eval scores for ranker $2 using file $4 test"
	echo " "
	echo "########################################"
	java -jar $RANK_LIB_FILE -load $1/models/$4_model.txt -test $dataRootPath/test_ranklib.txt -metric2T $EVAL_M -idv $1/predictions/eval_test.txt

	echo " "
	echo "########################################"
	echo " "
	echo "Eval scores for ranker $2 using file $4 train"
	echo " "
	echo "########################################"
	java -jar $RANK_LIB_FILE -load $1/models/$4_model.txt -test $dataRootPath/train_ranklib.txt -metric2T $EVAL_M -idv $1/predictions/eval_train.txt
	
}

function startExperiment {
	if [ ! -d "$1" ]; then
		createDirs "$1"
	fi
	runRanker $1 $2 $3 $4 ${@:5}
}

function evaluateExperiment {
  echo " "
	echo "########################################"
	echo " "
	echo "Eval scores"
	echo " "
	echo "########################################"
	java -jar $RANK_LIB_FILE -load $data_exp_path/R/models/R_model.txt -test $dataRootPath/R_test_ranklib.txt -metric2T $EVAL_M -idv $data_exp_path/R/predictions/eval_test.txt


	echo " "
	echo "########################################"
	echo " "
	echo "Predicting scores"
	echo " "
	echo "########################################"
	java -jar $RANK_LIB_FILE -load $data_exp_path/R/models/R_model.txt -rank $dataRootPath/R_test_ranklib.txt \
		-indri $data_exp_path/R/predictions/prediction.txt
}
#################################################
#						#
# Start of the complete routine			#
#			                        #
#################################################

LTR_SRC_DIR=$1
RANK_LIB_FILE="$LTR_SRC_DIR/RankLib-2.1-patched.jar" #/RankLib-2.12.jar"
METRIC="$2" 	#"ERR@50"		#"MAP"
RELEVANCE_LABEL=$4  # set number of relevance labels
TOP_K=$3
ranker="$5"				# read name of ranker from terminal
rankerID=$6				# read ranker ID from terminal
dataRootPath="$7"			# path to experimental dataset
data_exp_path="${10}"

EVAL_M="$METRIC@$TOP_K"
# THE LAST N ARGUMENTS OF THIS SCRIPT WILL BE HYPER PARAMETERS FOR TRAINING. 
# THEIR NUMBER CAN NOT BE SPECIFIED BEFORE RUNTIME BECAUSE DIFFERENT RANKERS NEED DIFFERENT PARAMETERS
# HENCE WE LOOP OVER ALL ARGUMENTS UNTIL WE COLLECTED THEM ALL

hyperParameters="${@:7}"

#for ((i=4;i<=$#;i++)); do echo "$#" "$i" "${!i}"; done

if [[ $dataRootPath == */ ]]; then
	dataRootPath="${dataRootPath::-1}"
fi

# write file about current training configuration
# writeInfoFile $ranker $hyperParameters

dateAndTime="experiments_$(date +%Y%m%d_%H%M%S)"

if true; then
trainingfile="$dataRootPath/'train_ranklib.txt"
echo $trainingfile

echo "DATA EXP PATH"
  echo $data_exp_path
    	# create new experiments dir
  if [ $data_exp_path == "none" ]; then
    echo "CREATE EXP DIR"
    exp_dir="$dataRootPath/$ranker/$dateAndTime/$timestamp"
    if [ ! -d "$exp_dir" ]; then
      echo "New experiments dir created."
      mkdir -p $exp_dir
      mkdir $exp_dir/models
      mkdir $exp_dir/predictions
    fi
  fi

# create info file
infofile="$exp_dir/info.txt"
echo $exp_dir
if [ -e $infofile ]; then
  echo "File $infofile already exists!"
else
  echo "Experiments properties " > $infofile
  echo "Ranker $ranker" >> $infofile
  echo "Ranker Parameters $hyperParameters" >> $infofile
  echo "Executed $(date +%d%m%Y) " >> $infofile
  echo "Relevance labels range: $RELEVANCE_LABEL" >> $infofile
  echo "Metric: $EVAL_M" >> $infofile
fi

    # start experiments and exit whole script if function returns -1
echo $data_exp_path
if [ $data_exp_path == "none" ]; then
  set -e
  echo $hyperParameters
  echo "---------"
  startExperiment $exp_dir $ranker $rankerID $timestamp $hyperParameters
  set +e
else
  set -e
  echo $hyperParameters
  evaluateExperiment
  echo "---------"
  set +e
fi

fi
