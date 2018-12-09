import fnmatch
import random, os
from PIL import Image

# Remove images smaller than 512 x 512
def file_filter(folder):
	for root, _, files in os.walk(folder):
   		for f in files:
   			fullpath = os.path.join(root, f)
			if fnmatch.fnmatch(f, '*.jpg') or fnmatch.fnmatch(f, '*.png') or fnmatch.fnmatch(f, '*.jpeg'):
				with Image.open(fullpath) as im:
					x, y = im.size
				if x < 512 or y < 512:
					print (fullpath)
					os.remove(fullpath)

# Remove images with file size smaller than 100k
def file_filter1(folder):
	for root, _, files in os.walk(folder):
		for f in files:
			fullpath = os.path.join(root, f)
			try:
				if os.path.getsize(fullpath) < 100 * 1024:   #set file size in kb
					print (fullpath)
					os.remove(fullpath)
			except WindowsError:
				print ("Error" + fullpath)

# Copy files
from shutil import copyfile

def cp(src, dst):
	for root, _, files in os.walk(src):
		for f in files:
			fullpath = os.path.join(root, f)
			copyfile(fullpath, os.path.join(dst, f))

# Split training/dev/test set
def split(folder, train, dev, test):
	total = 0
	for root, dirs, files in os.walk(folder):
		random.shuffle(files)
		for f in files:
			total += 1
			if total == 20000:
				return
				
			fullpath = os.path.join(root, f)
			
			if total < 10816:
				copyfile(fullpath, os.path.join(train, f))
			else :
				if total < 11072:
					copyfile(fullpath, os.path.join(dev, f))
				else :
					if total < 11328:
						copyfile(fullpath, os.path.join(test, f))

# Check number of PNG files
def CheckNO(folder):
	total = 0
	for root, dirs, files in os.walk(folder):
		for f in files:
			fullpath = os.path.join(root, f)
			if fnmatch.fnmatch(f, '*.png'):
				total += 1
	print (str(total))