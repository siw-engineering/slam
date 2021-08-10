GT_POSE=$1
POSE_EST=$2
README="../README.md"
READMEOUT="../README.md.out"
rm $READMEOUT

while IFS= read -r line
do
	if [ "$line" = "--- | --- | --- |" ];then
		echo "--- | --- | --- |" >> $READMEOUT
		echo "Score | "$(python2 evaluate_ate.py $GT_POSE $POSE_EST)" | "$(python2 evaluate_rpe.py $GT_POSE $POSE_EST)" |" >> $READMEOUT
		break
	else
		echo "$line" >> $READMEOUT
	fi


done < "$README"

rm $README
mv $READMEOUT $README