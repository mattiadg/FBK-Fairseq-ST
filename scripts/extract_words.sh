input=$1
sed "s/ //g" $input | sed "s/<space>/ /g" | cut -f3 > $input.word
