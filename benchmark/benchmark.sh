#run this file inside ./benchmark/
POSE_EST=$1
GT_POSE=$2
PLY_EST=$3
GT_PLY=$4

if [ -z $1 ] | [ -z $2 ] | [ -z $3 ] | [ -z $4 ]
then
      echo -e "cmd line args can't be empty!\nusage: ./benchmark --estimated_pose --ground_truth_pose --esitmated_ply --ground_truth_ply"
      exit
fi


README="../README.md"
READMEOUT="../README.md.out"
rm $READMEOUT
echo "building surfreg"
cd surfreg
mkdir -p build
cd build
cmake ../
make -j8
cd ../../

touch history.txt

echo "benchmarking ..."

while IFS= read -r line
do
	if [ "$line" = "--- | --- | --- | --- |" ];then
		echo "--- | --- | --- | --- |" >> $READMEOUT
		ATE=$(python2 evaluate_ate.py $GT_POSE $POSE_EST)
		RPE=$(python2 evaluate_rpe.py $GT_POSE $POSE_EST)
		PC_SCORE=$(./surfreg/build/SurfReg -r $PLY_EST -m $GT_PLY -t 0)
		echo "Score | "$ATE" | "$RPE" | "$PC_SCORE" | ">> $READMEOUT
		echo $(date)" "$ATE" "$RPE" "$PC_SCORE>> history.txt
		break
	else
		echo "$line" >> $READMEOUT
	fi


done < "$README"

rm $README
mv $READMEOUT $README