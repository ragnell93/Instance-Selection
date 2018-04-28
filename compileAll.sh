declare -a dataset=("iris" "cleveland" "led7digit" "pima" "wdbc" "monk" "wisconsin" "wine" "glass" "banknote"
"banana" "cardiotocography" "eye_state" "page_blocks" "penbased" "satimage" "thyroid" "segment"
"credit_card" "shuttle" "sensorless_drive")

for i in "${dataset[@]}"
do
    python3 mainEuclidean.py $i
done