# Compressed-Domain Video Object Tracking using Markov Random Fields with Graph Cuts Optimization

Research group at Fraunhofer HHI: <https://www.hhi.fraunhofer.de/en/departments/vca/research-groups/multimedia-communications.html>

The application performs video object tracking with compressed-domain processing of H.264/AVC video bitstreams applying Markov Random Fields and Graph Cuts methods.

---
## Compiling
##### Compile FFmpeg decoder for motion vector extraction
1. Clone repository into `libs` folder
```sh
cd libs
git clone https://github.com/bombardellif/FFmpeg.git
cd FFmpeg
mkdir bin
```
2. [Install dependencies and compile](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu) (assuming Linux Ubuntu)
```sh
sudo apt install autoconf \
    automake \
    build-essential \
    cmake \
    libass-dev \
    libfreetype6-dev \
    libtheora-dev \
    libtool \
    libvorbis-dev \
    mercurial \
    pkg-config \
    texinfo \
    zlib1g-dev \
    libx264-dev \
    yasm
PATH="./bin:$PATH" PKG_CONFIG_PATH="./ffmpeg_build/lib/pkgconfig" ./configure
--prefix=./ffmpeg_build --bindir=./bin --extra-cflags=-I./ffmpeg_build/include
--extra-ldflags=-L./ffmpeg_build/lib --enable-gpl --enable-libx264 --enablenonfree --enable-shared
PATH="./bin:$PATH" make -j4 && make install
```
##### Install application dependencies
1. Install Python and dependencies
```sh
sudo apt install python3-pip python3-dev python3-tk g++ gfortran liblapack-dev \
    liblapacke-dev libatlas-dev libopenblas-dev python3-cffi-backend \
    python3-cairo-dev libffi-dev python-opencv libopencv-dev
cd ../../
sudo pip3 install -r requirements.txt
```
2. Compile the sub modules
```sh
cd utils
python3 setup.py build_ext --inplace
cd ../mincut
python3 setup.py build_ext --inplace
cd ../decoder
python3 setup.py build_ext --inplace
```
3. Run the application (help output)
```sh
cd ..
LD_LIBRARY_PATH=libs/FFmpeg/ffmpeg_build/lib python3 -m hhi_stmrftracking.main -h
```

---
## Datasets

#### [VOT 2016 Challenge](http://www.votchallenge.net/vot2016/) Dataset

Download the pictures and ground truth at <http://www.votchallenge.net/vot2016/dataset.html>.

Copy the shell script `docs/encode-all.sh` inside the downloaded folder and execute it to encode the pictures in H.264/AVC.

#### Derf Dataset ([ST-MRF by Khatoonabadi and Bajić](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6272352&tag=1))

Download the ST-MRF code and dataset at <https://www.researchgate.net/publication/258938622_ST-MRF_tracking>. Videos are already encoded.

---
## Example
With the dataset copied to the `data` folder. The last parameter is the ground truth for the first picture of the video sequence.
```
LD_LIBRARY_PATH=libs/FFmpeg/ffmpeg_build/lib python3 -m hhi_stmrftracking.main -d data/in/gymnastics4.264 -e data/gt-gymnastics4/ data/in/gymnastics4.264 data/in/gt-gymnastics4.png
```
The expected output can be found in `docs/example_gymnastics4.264.eval.mp4`.
