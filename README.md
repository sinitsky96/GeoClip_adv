#GeoClip_adv

The environment was run using Python 3.11.11 in a Mamba environment with the libraries listed in requirements.txt.

To run the models for the project, only the variables that appear in the run_* bash scripts were used. While further variable optimization and experimentation is possible, we used the default settings from the papers' repositories we cloned.

Run examples are provided in the run_*.sh files.


All files for data downloads are present, use the functions in the download.py files in each of the dataset dirs to download relevant data.
exception for Im2GPS3k is made where you need to manually download the zip from https://www.mediafire.com/file/7ht7sn78q27o9we/im2gps3ktest.zip and extract it images to /data/Im2GPS3k/images.

modifications were made to both paper codes, SparseRS and PGDTrimKernel, where both needed to be adapted for the geolocation domain.<br>

SparseRS: was modified to allow multiple patch placements instead of one given $kernel area = \sqrt(k)\times \sqrt(k) < \epsilon$ ($L_0 limit$), in sparseRS, if we use sparse attack (kernel 1x1), k argument represents epsilon, for patches, epsilon is epsilon and k represent the AREA size of the kernel (k = pixels in each patch).