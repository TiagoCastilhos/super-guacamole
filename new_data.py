import tensorflow as tf
from tensorflow import keras

from main import image_size

img = keras.preprocessing.image.load_img(
    #"D:/Dev/tcc/catalogs-item-classification - legacy/ItemImages/incorretos/wbc_rt30x48_30r-0.png",
    #"D:/Dev/tcc/catalogs-item-classification - legacy/ItemImages/incorretos/bbc42r-8.png",
    #"D:/Dev/tcc/catalogs-item-classification - legacy/ItemImages/incorretos/wdcwl_tms24x48_12r-18.png",
    #"D:/Dev/tcc/catalogs-item-classification - legacy/ItemImages/corretos/2db24-27.png",
    #"D:/Dev/tcc/catalogs-item-classification - legacy/ItemImages/corretos/bls36l-0.png",
    #"C:/Users/tiago/Desktop/tcc/validar/corretas/22572__bhvbb_27_butt-6.png",
    #"C:/Users/tiago/Desktop/tcc/validar/corretas/22572__b_36_butt_2_dwr_top-3.png",
    #"C:/Users/tiago/Desktop/tcc/validar/incorretas/22653__bc_36_r-3.png",
    #"C:/Users/tiago/Desktop/tcc/validar/incorretas/22653__bhvb_15_l-6.png",
    "D:/Dev/tcc/super-guacamole/ItemImages/corretos/22572__2bowl_bhvbb_60-11.png",
    #"D:/Dev/Azure DevOps/item-thumbnail-generator/thumbnails/31398__b600/31398__b600-2.png",
    #"D:/Dev/Azure DevOps/item-thumbnail-generator/thumbnails/31395__wbc_lt42x30_12/31395__wbc_lt42x30_12-8.png",
    target_size=image_size
)

img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

reconstructed_model = keras.models.load_model("my_model.keras")
img.show("img")
predictions = reconstructed_model.predict(img_array)
score = predictions[0]
print(
    "This image is %.2f percent correct, %.2f percent incorrect and %.2f inconclusive."
    % (score[0] * 100, score[1] * 100, score[2] * 100)
)