install-linux:
	sudo pip install -r requirements.txt
	sudo apt-get install texlive-latex-base
	sudo apt-get install texlive-latex-extra

	sudo apt-get install ghostscript
	sudo apt-get install libgs-dev

	wget http://www.imagemagick.org/download/ImageMagick.tar.gz
	tar -xvf ImageMagick.tar.gz
	cd ImageMagick-7.*; \
	./configure --with-gslib=yes; \
	make; \
	sudo make install; \
	sudo ldconfig /usr/local/lib
	rm ImageMagick.tar.gz
	rm -r ImageMagick-7.*

install-mac:
	wget http://www.imagemagick.org/download/ImageMagick.tar.gz
	tar -xvf ImageMagick.tar.gz
	cd ImageMagick-7.*; \
	./configure --with-gslib=yes; \
	make;\
	sudo make install; \
	rm ImageMagick.tar.gz
	rm -r ImageMagick-7.*

run:
	python build.py
	python train.py
	python evaluate_txt.py
	python evaluate_img.py