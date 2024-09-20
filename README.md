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

To iterate through all the blocks one by one you could do

```
double_blocks\.0\.=1.1
double_blocks\.1\.=1.1
double_blocks\.2\.=1.1
double_blocks\.3\.=1.1
double_blocks\.4\.=1.1
double_blocks\.5\.=1.1
double_blocks\.6\.=1.1
double_blocks\.7\.=1.1
double_blocks\.8\.=1.1
double_blocks\.9\.=1.1
double_blocks\.10\.=1.1
double_blocks\.11\.=1.1
double_blocks\.12\.=1.1
double_blocks\.13\.=1.1
double_blocks\.14\.=1.1
double_blocks\.15\.=1.1
double_blocks\.16\.=1.1
double_blocks\.17\.=1.1
double_blocks\.18\.=1.1
single_blocks\.0\.=1.1
single_blocks\.1\.=1.1
single_blocks\.2\.=1.1
single_blocks\.3\.=1.1
single_blocks\.4\.=1.1
single_blocks\.5\.=1.1
single_blocks\.6\.=1.1
single_blocks\.7\.=1.1
single_blocks\.8\.=1.1
single_blocks\.9\.=1.1
single_blocks\.10\.=1.1
single_blocks\.11\.=1.1
single_blocks\.12\.=1.1
single_blocks\.13\.=1.1
single_blocks\.14\.=1.1
single_blocks\.15\.=1.1
single_blocks\.16\.=1.1
single_blocks\.17\.=1.1
single_blocks\.18\.=1.1
single_blocks\.19\.=1.1
single_blocks\.20\.=1.1
single_blocks\.21\.=1.1
single_blocks\.22\.=1.1
single_blocks\.23\.=1.1
single_blocks\.24\.=1.1
single_blocks\.25\.=1.1
single_blocks\.26\.=1.1
single_blocks\.27\.=1.1
single_blocks\.28\.=1.1
single_blocks\.29\.=1.1
single_blocks\.30\.=1.1
single_blocks\.31\.=1.1
single_blocks\.32\.=1.1
single_blocks\.33\.=1.1
single_blocks\.34\.=1.1
single_blocks\.35\.=1.1
single_blocks\.36\.=1.1
single_blocks\.37\.=1.1
```

The `Plot Block Params` node will then take the output of this iterator and plot the parameters directly onto the images.


## TODO

If there will be interest, I might add the following features:

- [ ] Speed up the patching process
- [ ] Support other models than Flux
- [ ] Make the patcher more user friendly, with a UI to edit the regex
- [ ] Better plotting of the blocks
