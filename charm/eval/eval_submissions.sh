<< COMMENT
This bash script evaluates all submissions in a given directory against the
ground truth using NIST's official evaluation script and saves them in the
specified output directory.

Usage: eval_submissions.sh <submission_dir> <ground_truth_dir> <scoring_index_filepath> <output_dir>

Example:
    $ stdbuf -o0 ./eval_submissions.sh ~/Documents/data/charm/transformed/predictions \
        ~/Documents/data/charm/raw/LDC2023E01_CCU_TA1_Mandarin_Chinese_Mini_Evaluation_Annotation_Unsequestered \
        ~/Documents/data/charm/raw/LDC2023E01_CCU_TA1_Mandarin_Chinese_Mini_Evaluation_Annotation_Unsequestered/index_files/COMPLETE.scoring.index.tab \
        ~/Documents/data/charm/transformed/scores
    NB: can optionally add stdbuf -o0 to the beginning of the command to see
    output as it is generated
COMMENT

# evaluate each submission in submision_dir
for submission in $1/*/; do
    # if submission is name "submissions", skip it
    if [ $(basename $submission) = "submissions" ]; then
        continue
    fi
    echo "Evaluating $submission"
    # create full output path
    mkdir -p $4/$(basename $submission)
    CCU_scoring score-cd -s $submission \
        -ref $2 \
        -i $3 \
        -o $4/$(basename $submission)
done 