# MaSA_UNet
We introduce the MaSA-UNet, a U-Net-like architecture complemented by the Manhattan Self-Attention mechanism for biomedical image segmentation. 


<p align="center">
  <figure>
    <img width="750" src="images/MaSA_UNet.png" alt="U-PEN Mamba Architecture">
    <figcaption>Detailed illustration of the proposed MaSA-UNet model. The picture depicts the key components of the architecture, the pre-trained MaSA denoised model, the MaSA segmentation model, and the WCL function.</figcaption>
  </figure>
</p>

## Usage
**Training the Model** (train_masaunet_segmentation_skin.py)
```
python -u train_masaunet_segmentation_skin.py --use_masa --use_autoencoder --dataset PH2
```


## License

See the [LICENSE](LICENSE.md) file for license rights and limitations (MIT).
