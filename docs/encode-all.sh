#! /bin/bash

seqs=`cat list.txt`
for seq in $seqs
do
    ffmpeg -y -f image2 -i ./$seq/%08d.jpg -pix_fmt yuv420p -vf "scale=iw:-2" -c:v libx264 -profile:v baseline ./$seq/$seq.264
done

# For the sequence gymnastics4
seq=gymnastics4
ffmpeg -y -f image2 -i ./$seq/%08d.jpg -pix_fmt yuv420p -vf "scale=-2:ih" -c:v libx264 -profile:v baseline ./$seq/$seq.264
# For the sequence soccer2
seq=soccer2
ffmpeg -y -f image2 -i ./$seq/%08d.jpg -pix_fmt yuv420p -vf "scale=iw+1:ih+1" -c:v libx264 -profile:v baseline ./$seq/$seq.264
