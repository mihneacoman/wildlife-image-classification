# 01 — Dataset Exploration

In this notebook, we perform an initial inspection of the selected wildlife camera-trap dataset subset with 1000 images per class.

The goal is to verify that the downloaded images and metadata are usable before moving to modeling. We check the class distribution, confirm that image paths are valid, inspect whether images can be opened correctly, and visualize random examples from each class.

This step helps identify possible issues such as corrupted files, incorrect labels, empty images, blurry samples, night images, or repeated frames.

The clean data is finally split into train/validation/test.


```python
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.notebook import tqdm

PROJECT_ROOT = Path.cwd().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"
SUBSET_PATH = METADATA_DIR / "subset_1000_per_class.csv"

print(PROJECT_ROOT)
print(SUBSET_PATH)
```

    /Users/mihnea/Desktop/Proiecte personale/wildlife-camera-trap-classification
    /Users/mihnea/Desktop/Proiecte personale/wildlife-camera-trap-classification/data/metadata/subset_1000_per_class.csv



```python
df = pd.read_csv(SUBSET_PATH)

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image_id</th>
      <th>file_name</th>
      <th>label</th>
      <th>category_id</th>
      <th>image_url</th>
      <th>local_path</th>
      <th>readable_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FL-16_03_07_2016_FL-16_0112146.JPG</td>
      <td>part1/sub190/FL-16_03_07_2016_FL-16_0112146.JPG</td>
      <td>sus scrofa</td>
      <td>62</td>
      <td>https://storage.googleapis.com/public-datasets...</td>
      <td>/Users/mihnea/Desktop/Proiecte personale/wildl...</td>
      <td>wild_boar</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CA-24_08_12_2015_CA-24_0009029.jpg</td>
      <td>part0/sub066/CA-24_08_12_2015_CA-24_0009029.jpg</td>
      <td>sus scrofa</td>
      <td>62</td>
      <td>https://storage.googleapis.com/public-datasets...</td>
      <td>/Users/mihnea/Desktop/Proiecte personale/wildl...</td>
      <td>wild_boar</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016_Unit028_SWTB009_img2060.jpg</td>
      <td>part0/sub025/2016_Unit028_SWTB009_img2060.jpg</td>
      <td>cervus elaphus</td>
      <td>10</td>
      <td>https://storage.googleapis.com/public-datasets...</td>
      <td>/Users/mihnea/Desktop/Proiecte personale/wildl...</td>
      <td>red_deer</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014_Unit97_Ivan093_img1677.jpg</td>
      <td>part0/sub017/2014_Unit97_Ivan093_img1677.jpg</td>
      <td>odocoileus hemionus</td>
      <td>46</td>
      <td>https://storage.googleapis.com/public-datasets...</td>
      <td>/Users/mihnea/Desktop/Proiecte personale/wildl...</td>
      <td>mule_deer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FL-37_06_03_2016_FL-37_0014913.JPG</td>
      <td>part2/sub278/FL-37_06_03_2016_FL-37_0014913.JPG</td>
      <td>procyon lotor</td>
      <td>54</td>
      <td>https://storage.googleapis.com/public-datasets...</td>
      <td>/Users/mihnea/Desktop/Proiecte personale/wildl...</td>
      <td>raccoon</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.DataFrame'>
    RangeIndex: 8000 entries, 0 to 7999
    Data columns (total 7 columns):
     #   Column          Non-Null Count  Dtype
    ---  ------          --------------  -----
     0   image_id        8000 non-null   str  
     1   file_name       8000 non-null   str  
     2   label           8000 non-null   str  
     3   category_id     8000 non-null   int64
     4   image_url       8000 non-null   str  
     5   local_path      8000 non-null   str  
     6   readable_label  8000 non-null   str  
    dtypes: int64(1), str(6)
    memory usage: 3.0 MB



```python
df["readable_label"].value_counts()
```




    readable_label
    wild_boar     1000
    red_deer      1000
    mule_deer     1000
    raccoon       1000
    coyote        1000
    black_bear    1000
    bobcat        1000
    empty         1000
    Name: count, dtype: int64




```python
df["local_path"] = df["local_path"].apply(lambda p: str(Path(p)))

df["exists"] = df["local_path"].apply(lambda p: Path(p).exists())

df["exists"].value_counts()
```




    exists
    True    8000
    Name: count, dtype: int64




```python
def can_open_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


df["can_open"] = df["local_path"].apply(can_open_image)

df["can_open"].value_counts()
```




    can_open
    True    8000
    Name: count, dtype: int64



## Class distribution is balanced by construction


```python
counts = df["readable_label"].value_counts().sort_index()

plt.figure(figsize=(10, 5))
counts.plot(kind="bar")
plt.title("Class distribution")
plt.xlabel("Class")
plt.ylabel("Number of images")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
```


    
![png](01_dataset_exploration_files/01_dataset_exploration_8_0.png)
    


## Visualizing sample images


```python
def show_image_grid(sample_df, image_col="local_path", label_col="readable_label", n_cols=4):
    n_images = len(sample_df)
    n_rows = (n_images + n_cols - 1) // n_cols

    plt.figure(figsize=(n_cols * 3, n_rows * 3))

    for idx, (_, row) in enumerate(sample_df.iterrows()):
        image = Image.open(row[image_col]).convert("RGB")

        ax = plt.subplot(n_rows, n_cols, idx + 1)
        ax.imshow(image)
        ax.set_title(row[label_col])
        ax.axis("off")

    plt.tight_layout()
    plt.show()


sample_df = df.sample(16, random_state=42)
show_image_grid(sample_df)
```


    
![png](01_dataset_exploration_files/01_dataset_exploration_10_0.png)
    



```python
for label in sorted(df["readable_label"].unique()):
    print(label)

    sample_df = (
        df[df["readable_label"] == label]
        .sample(8, random_state=42)
    )

    show_image_grid(sample_df, n_cols=4)
```

    black_bear



    
![png](01_dataset_exploration_files/01_dataset_exploration_11_1.png)
    


    bobcat



    
![png](01_dataset_exploration_files/01_dataset_exploration_11_3.png)
    


    coyote



    
![png](01_dataset_exploration_files/01_dataset_exploration_11_5.png)
    


    empty



    
![png](01_dataset_exploration_files/01_dataset_exploration_11_7.png)
    


    mule_deer



    
![png](01_dataset_exploration_files/01_dataset_exploration_11_9.png)
    


    raccoon



    
![png](01_dataset_exploration_files/01_dataset_exploration_11_11.png)
    


    red_deer



    
![png](01_dataset_exploration_files/01_dataset_exploration_11_13.png)
    


    wild_boar



    
![png](01_dataset_exploration_files/01_dataset_exploration_11_15.png)
    



```python
clean_df = df[df["exists"] & df["can_open"]].copy()

clean_df.shape
```




    (8000, 9)



## Inspect Image Sizes


```python
def get_image_size(path):
    with Image.open(path) as img:
        return img.size  # width, height


sizes = clean_df["local_path"].apply(get_image_size)

clean_df["width"] = sizes.apply(lambda x: x[0])
clean_df["height"] = sizes.apply(lambda x: x[1])

clean_df[["width", "height"]].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>width</th>
      <th>height</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8000.000000</td>
      <td>8000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2049.204000</td>
      <td>1534.908000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>42.662137</td>
      <td>38.117045</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1920.000000</td>
      <td>1080.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2048.000000</td>
      <td>1536.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2048.000000</td>
      <td>1536.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2048.000000</td>
      <td>1536.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3648.000000</td>
      <td>2736.000000</td>
    </tr>
  </tbody>
</table>
</div>



Most images have a resolution of 2048×1536, with only a small number of images having different dimensions.

## Cropping text from the images

Near the top and bottom edge of the image, there is some text containing information about the picture (date/time, camera, temperature).
We want to focus only on the actual scene, so we crop the text.

Based on visual inspection, we decided to crop the top and bottom `30px`.


```python
CROPPED_DIR = DATA_DIR / "processed" / "cropped_images"
CROPPED_DIR.mkdir(parents=True, exist_ok=True)

TOP_CROP = 30
BOTTOM_CROP = 30
```


```python
def crop_camera_trap_borders(image, top_crop=30, bottom_crop=30):
    """
    Crop fixed top and bottom borders from a camera-trap image.

    The original image is not modified.
    """
    width, height = image.size

    left = 0
    upper = top_crop
    right = width
    lower = height - bottom_crop

    if lower <= upper:
        raise ValueError(
            f"Invalid crop for image with height={height}: "
            f"top_crop={top_crop}, bottom_crop={bottom_crop}"
        )

    return image.crop((left, upper, right, lower))
```


```python
def show_original_vs_cropped(path, top_crop=30, bottom_crop=30):
    original = Image.open(path).convert("RGB")
    cropped = crop_camera_trap_borders(
        original,
        top_crop=top_crop,
        bottom_crop=bottom_crop,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(cropped)
    axes[1].set_title("Cropped")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
```


```python
sample_df = clean_df.sample(10, random_state=42)

for _, row in sample_df.iterrows():
    print(row["readable_label"])
    show_original_vs_cropped(
        row["local_path"],
        top_crop=TOP_CROP,
        bottom_crop=BOTTOM_CROP,
    )
```

    mule_deer



    
![png](01_dataset_exploration_files/01_dataset_exploration_20_1.png)
    


    black_bear



    
![png](01_dataset_exploration_files/01_dataset_exploration_20_3.png)
    


    raccoon



    
![png](01_dataset_exploration_files/01_dataset_exploration_20_5.png)
    


    black_bear



    
![png](01_dataset_exploration_files/01_dataset_exploration_20_7.png)
    


    bobcat



    
![png](01_dataset_exploration_files/01_dataset_exploration_20_9.png)
    


    red_deer



    
![png](01_dataset_exploration_files/01_dataset_exploration_20_11.png)
    


    red_deer



    
![png](01_dataset_exploration_files/01_dataset_exploration_20_13.png)
    


    mule_deer



    
![png](01_dataset_exploration_files/01_dataset_exploration_20_15.png)
    


    coyote



    
![png](01_dataset_exploration_files/01_dataset_exploration_20_17.png)
    


    black_bear



    
![png](01_dataset_exploration_files/01_dataset_exploration_20_19.png)
    



```python
sample_df = clean_df.sample(20, random_state=123)

for _, row in sample_df.iterrows():
    print(row["readable_label"])
    show_original_vs_cropped(
        row["local_path"],
        top_crop=TOP_CROP,
        bottom_crop=BOTTOM_CROP,
    )
```

    raccoon



    
![png](01_dataset_exploration_files/01_dataset_exploration_21_1.png)
    


    raccoon



    
![png](01_dataset_exploration_files/01_dataset_exploration_21_3.png)
    


    coyote



    
![png](01_dataset_exploration_files/01_dataset_exploration_21_5.png)
    


    mule_deer



    
![png](01_dataset_exploration_files/01_dataset_exploration_21_7.png)
    


    black_bear



    
![png](01_dataset_exploration_files/01_dataset_exploration_21_9.png)
    


    coyote



    
![png](01_dataset_exploration_files/01_dataset_exploration_21_11.png)
    


    black_bear



    
![png](01_dataset_exploration_files/01_dataset_exploration_21_13.png)
    


    coyote



    
![png](01_dataset_exploration_files/01_dataset_exploration_21_15.png)
    


    black_bear



    
![png](01_dataset_exploration_files/01_dataset_exploration_21_17.png)
    


    bobcat



    
![png](01_dataset_exploration_files/01_dataset_exploration_21_19.png)
    


    empty



    
![png](01_dataset_exploration_files/01_dataset_exploration_21_21.png)
    


    red_deer



    
![png](01_dataset_exploration_files/01_dataset_exploration_21_23.png)
    


    black_bear



    
![png](01_dataset_exploration_files/01_dataset_exploration_21_25.png)
    


    black_bear



    
![png](01_dataset_exploration_files/01_dataset_exploration_21_27.png)
    


    red_deer



    
![png](01_dataset_exploration_files/01_dataset_exploration_21_29.png)
    


    bobcat



    
![png](01_dataset_exploration_files/01_dataset_exploration_21_31.png)
    


    wild_boar



    
![png](01_dataset_exploration_files/01_dataset_exploration_21_33.png)
    


    red_deer



    
![png](01_dataset_exploration_files/01_dataset_exploration_21_35.png)
    


    coyote



    
![png](01_dataset_exploration_files/01_dataset_exploration_21_37.png)
    


    black_bear



    
![png](01_dataset_exploration_files/01_dataset_exploration_21_39.png)
    



```python
def get_cropped_path(original_path):
    original_path = Path(original_path)

    relative_path = original_path.relative_to(DATA_DIR / "raw" / "images")

    return CROPPED_DIR / relative_path
```


```python
def save_cropped_images(df, top_crop=30, bottom_crop=30):
    cropped_paths = []

    for path in tqdm(df["local_path"], desc="Saving cropped images"):
        original_path = Path(path)
        cropped_path = get_cropped_path(original_path)

        cropped_path.parent.mkdir(parents=True, exist_ok=True)

        if not cropped_path.exists():
            image = Image.open(original_path).convert("RGB")
            cropped = crop_camera_trap_borders(
                image,
                top_crop=top_crop,
                bottom_crop=bottom_crop,
            )
            cropped.save(cropped_path)

        cropped_paths.append(str(cropped_path))

    return cropped_paths
```


```python
clean_df["cropped_path"] = save_cropped_images(
    clean_df,
    top_crop=TOP_CROP,
    bottom_crop=BOTTOM_CROP,
)
```

### Verify cropped images.


```python
clean_df["cropped_exists"] = clean_df["cropped_path"].apply(lambda p: Path(p).exists())

clean_df["cropped_exists"].value_counts()
```




    cropped_exists
    True    8000
    Name: count, dtype: int64




```python
def get_image_size(path):
    with Image.open(path) as img:
        return img.size

cropped_sizes = clean_df["cropped_path"].apply(get_image_size)

clean_df["cropped_width"] = cropped_sizes.apply(lambda x: x[0])
clean_df["cropped_height"] = cropped_sizes.apply(lambda x: x[1])

clean_df[["cropped_width", "cropped_height"]].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cropped_width</th>
      <th>cropped_height</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8000.000000</td>
      <td>8000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2049.204000</td>
      <td>1474.908000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>42.662137</td>
      <td>38.117045</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1920.000000</td>
      <td>1020.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2048.000000</td>
      <td>1476.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2048.000000</td>
      <td>1476.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2048.000000</td>
      <td>1476.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3648.000000</td>
      <td>2676.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Save final metadata


```python
clean_df["local_path_rel"] = clean_df["local_path"].apply(
    lambda p: str(Path(p).relative_to(PROJECT_ROOT))
)
clean_df["cropped_path_rel"] = clean_df["cropped_path"].apply(
    lambda p: str(Path(p).relative_to(PROJECT_ROOT))
)

clean_path = METADATA_DIR / "subset_1000_per_class_clean.csv"
clean_df.to_csv(clean_path, index=False)

clean_path
```




    PosixPath('/Users/mihnea/Desktop/Proiecte personale/wildlife-camera-trap-classification/data/metadata/subset_1000_per_class_clean.csv')



## Dataset split (train/validation/test)

We split the cleaned dataset into train, validation, and test sets. The metadata contains both original and cropped image paths, so later notebooks can choose which version to use.


```python
from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(
    clean_df,
    train_size=0.70,
    stratify=clean_df["readable_label"],
    random_state=42,
)

val_df, test_df = train_test_split(
    temp_df,
    train_size=0.50,
    stratify=temp_df["readable_label"],
    random_state=42,
)

train_df.to_csv(METADATA_DIR / "train_1000.csv", index=False)
val_df.to_csv(METADATA_DIR / "val_1000.csv", index=False)
test_df.to_csv(METADATA_DIR / "test_1000.csv", index=False)

splits = {
    "Train": train_df,
    "Validation": val_df,
    "Test": test_df,
}

for split_name, split_df in splits.items():
    print(f"{split_name} shape: {split_df.shape}")
    print(split_df["readable_label"].value_counts().sort_index())
    print()
```

    Train shape: (5600, 15)
    readable_label
    black_bear    700
    bobcat        700
    coyote        700
    empty         700
    mule_deer     700
    raccoon       700
    red_deer      700
    wild_boar     700
    Name: count, dtype: int64
    
    Validation shape: (1200, 15)
    readable_label
    black_bear    150
    bobcat        150
    coyote        150
    empty         150
    mule_deer     150
    raccoon       150
    red_deer      150
    wild_boar     150
    Name: count, dtype: int64
    
    Test shape: (1200, 15)
    readable_label
    black_bear    150
    bobcat        150
    coyote        150
    empty         150
    mule_deer     150
    raccoon       150
    red_deer      150
    wild_boar     150
    Name: count, dtype: int64
    


## Final dataset summary

The cleaned dataset contains 8000 images across 8 balanced classes (1000 images per class). All files were verified to exist and open successfully. Cropped copies were created in the processed data directory while keeping the original images unchanged. Stratified train, validation, and test splits were saved for modeling.
