sed -i "s/ /*/g" $1

python $(dirname $0)/space_char.py $1 $1".tmp"

rm $1

mv $1".tmp" $1

sed -i "s/*/<space>/g" $1


