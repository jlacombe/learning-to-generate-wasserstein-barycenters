#!/bin/bash
# usage: ./run_interpolate_XP.sh [model_id] [results_XPs_path] + comment/uncomment desired lines

echo "================================"
echo "== INTERPOLATIONS EXPERIMENTS =="
echo "================================"
triangle_size=2 # min=1
pentagon_size=3 # min=2

echo "Generating barycenters - same classes, triangle ..."
python interpolate_XP.py -m $1 -r $2 -k "triangle" -f $4 -l "cat1,cat2,cat3" -t $5 -i "cats" -n $triangle_size -c $3
echo "Generating barycenters - same classes, pentagon ..."
python interpolate_XP.py -m $1 -r $2 -k "pentagon" -f $4 -l "cat1,cat2,cat3,cat4,cat5" -t $5 -i "cats" -n $pentagon_size -c $3
echo "Generating barycenters - same classes, line ..."
python interpolate_XP.py -m $1 -r $2 -k "line" -f $4 -l "cat1,cat2" -t $5 -i "cats_28x28" -w "0.0,1.0|0.1,0.9|0.2,0.8|0.3,0.7|0.4,0.6|0.5,0.5|0.6,0.4|0.7,0.3|0.8,0.2|0.9,0.1|1.0,0.0" -c $3 -s '28,28'
python interpolate_XP.py -m $1 -r $2 -k "line" -f $4 -l "cat1,cat2,cat3" -t $5 -i "cats" -n 8 -c $3
python interpolate_XP.py -m $1 -r $2 -k "line" -f $4 -l "cat1,cat2,cat3,cat4,cat5" -t $5 -i "cats" -n 8 -c $3
python interpolate_XP.py -m $1 -r $2 -k "line" -f $4 -l "cat1,cat2" -t $5 -i "cats_iso" -w "0.5,0.5|0.5,0.5" -c $3

echo "Generating barycenters - different classes, triangle ..."
python interpolate_XP.py -m $1 -r $2 -k "triangle" -f $4 -l "cat1,car1,cloud1" -t $5 -i "diff" -n $triangle_size -c $3
echo "Generating barycenters - different classes, pentagon ..."
python interpolate_XP.py -m $1 -r $2 -k "pentagon" -f $4 -l "cat1,car1,cloud1,diamond1,owl1" -t $5 -i "diff" -n $pentagon_size -c $3
echo "Generating barycenters - different classes, line ..."
python interpolate_XP.py -m $1 -r $2 -k "line" -f $4 -l "car1,owl1" -t $5 -i "diff" -w "0.0,1.0|0.1,0.9|0.2,0.8|0.3,0.7|0.4,0.6|0.5,0.5|0.6,0.4|0.7,0.3|0.8,0.2|0.9,0.1|1.0,0.0" -c $3
python interpolate_XP.py -m $1 -r $2 -k "line" -f $4 -l "car1,owl1" -t $5 -i "diff_28x28" -w "0.0,1.0|0.1,0.9|0.2,0.8|0.3,0.7|0.4,0.6|0.5,0.5|0.6,0.4|0.7,0.3|0.8,0.2|0.9,0.1|1.0,0.0" -c $3 -s '28,28'
python interpolate_XP.py -m $1 -r $2 -t "line" -f $4 -l "cat1,car1,cloud1" -t $5 -i "diff" -n 8 -c $3
python interpolate_XP.py -m $1 -r $2 -t "line" -f $4 -l "cat1,car1,cloud1,diamond1,owl1" -t $5 -i "diff" -n 8 -c $3

echo "Generating interpolation of lines..."
python interpolate_XP.py -m $1 -r $2 -k "pentagon" -f $4 -l "line1,line2,line3,line4,line5" -t $5 -i "fig4" -n $pentagon_size -c $3
python interpolate_XP.py -m $1 -r $2 -k "line" -f $4 -l "line1,line2,line3,line4,line5" -t $5 -i "lines" -n 8 -c $3
python interpolate_XP.py -m $1 -r $2 -k "line" -f $4 -l "line4,line5" -t $5 -i "lines" -w "0.0,1.0|0.1,0.9|0.2,0.8|0.3,0.7|0.4,0.6|0.5,0.5|0.6,0.4|0.7,0.3|0.8,0.2|0.9,0.1|1.0,0.0" -c $3

echo "Generating interpolation of ellipses..."
python interpolate_XP.py -m $1 -r $2 -k "pentagon" -f $4 -l "circle1,circle2,circle3,circle4,circle5" -t $5 -i "fig6" -n $pentagon_size -c $3
python interpolate_XP.py -m $1 -r $2 -k "line" -f $4 -l "circle1,circle2,circle3,circle4,circle5" -t $5 -i "ellipses" -n 8 -c $3
python interpolate_XP.py -m $1 -r $2 -k "line" -f $4 -l "circle3,circle15" -t $5 -i "circles" -w "0.0,1.0|0.1,0.9|0.2,0.8|0.3,0.7|0.4,0.6|0.5,0.5|0.6,0.4|0.7,0.3|0.8,0.2|0.9,0.1|1.0,0.0" -c $3

echo "Generating barycenter of 100 cats..."
python interpolate_XP.py -m $1 -r $2 -k "line" -f $4 -l "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100" -t $5 -i "catstest" -w "0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01" -c $3

# echo "Generating barycenters - full shapes, triangle ..."
# python interpolate_XP.py -m $1 -r $2 -k "triangle" -f $4 -l "double_disk,heart,cross" -t $5 -i "full" -n $triangle_size -c $3
# echo "Generating barycenters - full shapes, pentagon ..."
# python interpolate_XP.py -m $1 -r $2 -k "pentagon" -f $4 -l "double_disk,heart,cross,duck,tooth" -t $5 -i "full" -n $pentagon_size -c $3
# echo "Generating barycenters - full shapes, line ..."
# python interpolate_XP.py -m $1 -r $2 -k "line" -f $4 -l "double_disk,heart,cross" -t $5 -i "full" -n 6 -c $3
# python interpolate_XP.py -m $1 -r $2 -k "line" -f $4 -l "double_disk,heart,cross,duck,tooth" -t $5 -i "full" -n 6 -c $3

echo "Building interpolation animations..."
python interpolate_XP.py -m $1 -r $2 -k "anim" -f $4 -l "cat1,cat2" -t $5 -c $3
python interpolate_XP.py -m $1 -r $2 -k "anim" -f $4 -l "car1,owl1" -t $5 -c $3
# with full shapes:
# python interpolate_XP.py -m $1 -r $2 -k "anim" -f $4 -l "double_disk,duck" -t $5 -c $3
# python interpolate_XP.py -m $1 -r $2 -k "anim" -f $4 -l "double_disk,heart" -t $5 -c $3

