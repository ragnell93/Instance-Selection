declare -a small=("iris" "monk")
declare -a medium=("banana" "penbased")

declare -a heuristics=("cnn")
declare -a meta=("chc")

for i in "${small[@]}"
do
    for j in "${heuristics[@]}"
    do
        for k in "${heuristics[@]}"
        do
            for p in "${meta[@]}"
            do
                ./main $i euclidean ./config/small $j $k $p  
                echo "Done with $i $j $k $p" 
            done
        done
    done
done

for i in "${medium[@]}"
do
    for j in "${heuristics[@]}"
    do
        for k in "${heuristics[@]}"
        do
            for p in "${meta[@]}"
            do
                ./main $i euclidean ./config/medium $j $k $p 
                echo "Done with $i $j $k $p"
            done
        done
    done
done