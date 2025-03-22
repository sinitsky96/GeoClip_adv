wget https://github.com/TIBHannover/GeoEstimation/releases/download/pytorch/yfcc25600_urls.csv -O $1/yfcc25600_urls.csv
# python download_images.py --output $1/images/yfcc25600 --url_csv $1/yfcc25600_urls.csv --shuffle --size_suffix ""
wget https://github.com/TIBHannover/GeoEstimation/releases/download/pytorch/yfcc25600_places365.csv -O $1/yfcc25600_places365.csv
# python $1/assign_classes.py
# python $1/filter_by_downloaded_images.py
