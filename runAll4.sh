declare -a small=("iris" "cleveland" "led7digit" "pima" "wdbc" "monk2" "wisconsin" "wine" "glass" "banknote" 
                  "appendicitis" "balance" "bands" "contraceptive" "dermatology" "ecoli" "haberman" "hayes_roth"
                  "heart" "hepatitis" "mammographic" "newthyroid" "tae" "vehicle" "vowel" "yeast")
declare -a medium=("banana" "cardiotocography" "eye_state" "page_blocks" "penbased" "satimage" "thyroid" "segment"
                    "coil2000" "magic" "marketing" "phoneme" "ring" "spambase" "texture" "titanic" "twonorm" )
declare -a large=("credit_card" "shuttle")

for i in "${small[@]}"
do

    ./main -i ./Instances/$i --mh gga --crossr 0.4837 --mutr 0.0001 --tsize 1 --pop 70 --iter 1000 --distance euclidean --strat 0 --nfolds 10 --k 1 --cost 0.5 --pini 0.3 --h1 cnn --h2 rss  
    echo "Done with $i cnn rss gga"

    ./main -i ./Instances/$i --mh gga --crossr 0.4837 --mutr 0.0001 --tsize 1 --pop 70 --iter 1000 --distance euclidean --strat 0 --nfolds 10 --k 1 --cost 0.5 --pini 0.3 --h1 enn --h2 rss  
    echo "Done with $i enn rss gga"


done

for i in "${medium[@]}"
do
    ./main -i ./Instances/$i --mh gga --crossr 0.5779 --mutr 0.0001 --tsize 1 --pop 88 --iter 1000 --distance euclidean --strat 1 --nfolds 10 --k 1 --cost 0.5 --pini 0.3 --h1 cnn --h2 rss  
    echo "Done with $i cnn rss gga"

    ./main -i ./Instances/$i --mh gga --crossr 0.5779 --mutr 0.0001 --tsize 1 --pop 88 --iter 1000 --distance euclidean --strat 1 --nfolds 10 --k 1 --cost 0.5 --pini 0.3 --h1 enn --h2 rss  
    echo "Done with $i enn rss gga"


done

for i in "${large[@]}"
do

    ./main -i ./Instances/$i --mh gga --crossr 0.5158 --mutr 0.0001 --tsize 1 --pop 102 --iter 1000 --distance euclidean --strat 1 --nfolds 50 --k 1 --cost 0.5 --pini 0.3 --h1 cnn --h2 rss  
    echo "Done with $i cnn rss gga"

    ./main -i ./Instances/$i --mh gga --crossr 0.5158 --mutr 0.0001 --tsize 1 --pop 102 --iter 1000 --distance euclidean --strat 1 --nfolds 50 --k 1 --cost 0.5 --pini 0.3 --h1 enn --h2 rss  
    echo "Done with $i enn rss gga"





done
