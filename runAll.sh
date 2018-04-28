declare -a small=("iris" "cleveland" "led7digit" "pima" "wdbc" "monk" "wisconsin" "wine" "glass" "banknote")
declare -a medium=("banana" "cardiotocography" "eye_state" "page_blocks" "penbased" "satimage" "thyroid" "segment")
declare -a large=("credit_card" "shuttle")

declare -a smallConfig=("small_cnn_genetic" "small_rss_genetic" "small_ib3_genetic")
declare -a mediumConfig=("medium_cnn_genetic" "medium_rss_genetic" "medium_ib3_genetic")
declare -a largeConfig=("large_cnn_genetic" "large_rss_genetic" "large_ib3_genetic")

for i in "${small[@]}"
do
    for j in "${smallConfig[@]}"
    do
        ./main $i euclidean ./config/$j
        echo "Done with $i $j" 
    done
done

for i in "${medium[@]}"
do
    for j in "${mediumConfig[@]}"
    do
        ./main $i euclidean ./config/$j
        echo "Done with $i $j" 
    done
done

for i in "${large[@]}"
do
    for j in "${largeConfig[@]}"
    do
        ./main $i euclidean ./config/$j
        echo "Done with $i $j" 
    done
done
