declare -a dataset=("iris" "cleveland" "led7digit" "pima" "wdbc" "monk" "wisconsin" "wine" "glass" "banknote"
"banana" "cardiotocography" "eye_state" "page_blocks" "penbased" "satimage" "thyroid" "segment"
"credit_card" "shuttle" "sensorless_drive" "appendicitis" "balance"  "bands" "coil2000" "contraceptive" "dermatology" 
"ecoli" "haberman" "hayes_roth" "heart" "hepatitis" "magic" "mammographic" "marketing" "newthyroid" "phoneme" "ring" 
"spambase" "tae" "texture" "titanic" "twonorm" "vehicle" "vowel" "yeast" )

for i in "${dataset[@]}"
do
    python3 mainEuclidean.py $i
done