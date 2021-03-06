declare -a small=("iris" "cleveland" "led7digit" "pima" "wdbc" "monk2" "wisconsin" "wine" "glass" "banknote" 
                  "appendicitis" "balance" "bands" "contraceptive" "dermatology" "ecoli" "haberman" "hayes_roth"
                  "heart" "hepatitis" "mammographic" "newthyroid" "tae" "vehicle" "vowel" "yeast")
declare -a medium=("banana" "cardiotocography" "eye_state" "page_blocks" "penbased" "satimage" "thyroid" "segment"
                    "coil2000" "magic" "marketing" "phoneme" "ring" "spambase" "texture" "titanic" "twonorm" )
declare -a large=("credit_card" "shuttle")

declare -a heuristics=("cnn")

for i in "${small[@]}"
do
    for j in "${heuristics[@]}"
    do
            ./main -i ./Instances/$i --mh gga --crossr 0.9463 --mutr 0.0001 --tsize 1 --pop 70 --iter 1000 --distance euclidean --strat 0 --nfolds 10 --k 1 --cost 0.5 --pini 0.3 --h1 cnn --h2 cnn  
            echo "Done with $i cnn cnn gga"

            ./main -i ./Instances/$i --mh sga --iter 1000 --pop 90 --crossr 0.9848 --mutr 0.0057 --tsize 3 --distance euclidean --strat 0 --nfolds 10 --k 1 --cost 0.5 --pini 0.3 --h1 $j --h2 $j  
            echo "Done with $i $j $j sga"

    done
done

for i in "${medium[@]}"
do
    for j in "${heuristics[@]}"
    do

            ./main -i ./Instances/$i --mh gga --crossr 0.9513 --mutr 0.0001 --tsize 1 --pop 88 --iter 1000 --distance euclidean --strat 1 --nfolds 10 --k 1 --cost 0.5 --pini 0.3 --h1 cnn --h2 cnn  
            echo "Done with $i cnn cnn gga"

            ./main -i ./Instances/$i --mh sga --iter 1000 --pop 132 --crossr 0.9859 --mutr 0.0001 --tsize 1 --distance euclidean --strat 1 --nfolds 10 --k 1 --cost 0.5 --pini 0.3 --h1 $j --h2 $j  
            echo "Done with $i $j $j sga"

    done
done

for i in "${large[@]}"
do
    for j in "${heuristics[@]}"
    do

            ./main -i ./Instances/$i --mh gga --crossr 0.9482 --mutr 0.0001 --tsize 1 --pop 102 --iter 1000 --distance euclidean --strat 1 --nfolds 50 --k 1 --cost 0.5 --pini 0.3 --h1 cnn --h2 cnn  
            echo "Done with $i cnn cnn gga"

            ./main -i ./Instances/$i --mh sga --iter 1000 --pop 122 --crossr 0.9554 --mutr 0.0078 --tsize 7 --distance euclidean --strat 1 --nfolds 50 --k 1 --cost 0.5 --pini 0.3 --h1 $j --h2 $j  
            echo "Done with $i $j $j sga"

    done
done
