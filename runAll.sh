declare -a small=("iris" "cleveland" "led7digit" "pima" "wdbc" "monk" "wisconsin" "wine" "glass" "banknote")
declare -a medium=("banana" "cardiotocography" "eye_state" "page_blocks" "penbased" "satimage" "thyroid" "segment")
declare -a large=("credit_card" "shuttle")

declare -a heuristics=("cnn")
declare -a meta=("genetic" "memetic" "chc")

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


for i in "${large[@]}"
do
    for j in "${heuristics[@]}"
    do
        for k in "${heuristics[@]}"
        do
            for p in "${meta[@]}"
            do
                ./main $i euclidean ./config/large $j $k $p  
                echo "Done with $i $j $k $p"
            done
        done
    done
done
