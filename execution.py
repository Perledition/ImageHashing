# import standard modules
import glob

# import third party modules
import imagehash
import numpy as np
import pandas as pd
from scipy.spatial.distance import hamming

# import project related modules
from average_hashing import ImageHashing

# https://www.pyimagesearch.com/2019/08/26/building-an-image-hashing-search-engine-with-vp-trees-and-opencv/
# https://www.youtube.com/watch?v=61pUX7_LKZg  - CBIR


def sample_images(image_path: str, seed: int = 2, n_groups: int = 3, n_images: int = 5):
    """
    Computes a disproportionate stratified random sample of images

    :param image_path: path containing all images
    :param seed: seed setting for random group & image draw for reproducability (default value: 2)
    :param n_groups: number of strata from which to draw n_images respectively (default value: 2)
    :param n_images: number of images to randomly sample from each strata (default value: 5)

    :return disproportionate stratified random sample of images
    """

    # Fetch all image strings in directory
    image_df = pd.DataFrame(glob.glob(f"{image_path}/*.jpg"), columns=["Image_Name"])

    # assemble dataframe
    image_df["Image_Name"] = image_df["Image_Name"].map(lambda x: x.replace('images/', ''))
    image_df["Image_Group"] = image_df["Image_Name"].map(lambda x: x.replace(x[4:], ''))

    # adjust column sorting
    image_df = image_df.reindex(["Image_Group", "Image_Name"], axis=1)

    # set seed for reproducability
    np.random.seed(seed)

    # create grouped instance
    image_df_grouped = image_df.groupby("Image_Group")

    # derive array containing all groups
    image_groups_array = np.arange(image_df_grouped.ngroups)

    # randomly shuffle array
    np.random.shuffle(image_groups_array)

    # subset original df through element-wise membership check against subset of randomly shuffled array
    image_df = image_df.loc[image_df_grouped.ngroup().isin(image_groups_array[:n_groups])]

    # randomly sample n Images per group
    image_sub_df = image_df.groupby('Image_Group').apply(lambda x: x.sample(n_images, replace=False, random_state=seed))
    image_sub_df.reset_index(drop=True, inplace=True)

    return image_sub_df


# define path
path = "C:\\Users\\Maxim\\Dropbox\\Fotos"

# create instance hasher
hasher = ImageHashing()

# Generate image sample
# image_sub_df = sample_images(path, seed=2, n_groups=3, n_images=5)
image_sub_df = pd.DataFrame(glob.glob(f"{path}/*.jpg"), columns=["Image_Name"])
print(image_sub_df.head())

# create empty df with default column array
col_list = image_sub_df.columns.tolist()

hash_list = dict()

# iterate over image sample to compute hashes from each candidate algorithm
for index_row, image in enumerate(image_sub_df.Image_Name):
    hash_value = hasher.from_file(f"{image}")

    print(image, hash_value.hash)
    hash_list[image] = hash_value.hash

# search for similar images given the example
# hash_ = imagehash.dhash(Image.open(f"{search_path}/{image}"), hash_size=16)  .hash.reshape(hash_size**2,)*1
# hash =  imagehash.average_hash(Image.open(f"{search_path}/{image}"), hash_size=16)  .hash.reshape(hash_size**2,)*1


# make a quick cross check, this is something you would query with something more sophisticated
for image in hash_list.keys():
    key_hash = hash_list[image]
    for key, value in hash_list.items():
        hamming_distance = hamming(key_hash, value)
        print(f"{image}  -> {key}: {hamming_distance}")