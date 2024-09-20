# Flux blocks patcher sampler

This is an (very) advanced and (very) experimental custom node for the ComfyUI. It allows you to iteratively change the blocks weights of Flux models and check the difference each value makes.

## Usage

The `blocks` parameter accepts a list of blocks you want to iterate over, one per line. Each line should have the format `regex=weight`. For example the following lines will iterate over all the weights in the double_blocks first and then the single_blocks:

```
double_blocks\.([0-9]+)\.(img|txt)_(mod|attn|mlp\.[02])\.(lin|qkv|proj)\.(weight|bias)=1.1
single_blocks\.([0-9]+)\.(linear[12]|modulation\.lin)\.(weight|bias)=1.1
```

The regex above shows all the block options you have, but it can be as complex or simple as you want. For example if you want to target the whole `double_blocks 0`, you can use something like:

```
double_blocks\.0\.=1.1
```

To patch the `img` weights only of all double blocks you can use the following:

```
double_blocks\.([0-9]+)\.img_=1.1
```

To patch all the single blocks you can use:

```
single_blocks=1.1
```

The `Block Params Plot` node will then take the output of this iterator and plot the parameters directly onto the images.


## TODO

If there will be interest, I might add the following features:

- [ ] Speed up the patching process
- [ ] Support other models than Flux
- [ ] Make the patcher more user friendly, with a UI to edit the regex
- [ ] Better plotting of the blocks
