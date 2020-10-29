# FFPolish - Filters Artifacts From Formalin-Fixed Paraffin-Embedded (FFPE) Variants

## Installation 
### Conda Installation
Ensure that you have the `conda-forge` and `bioconda` channels in the correct priority order.

```
conda config --show channels
channels:
  - conda-forge
  - bioconda
  - defaults
```

If the above command doesn't have the correct output, run:
```
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
```

To create a new environment named `ffpolish` with FFPolish installed, run:
```
conda create -n ffpolish -c matnguyen ffpolish
```

Activate the environment 
```
source activate ffpolish
```

And run FFPolish, producing the help output
```
ffpolish -h
```

## Running FFPolish
### Filtering VCF
You can filter artifacts from a VCF using the pre-trained model. 

#### Input Requirements
* A reference genome in FASTA format
* A bgzipped VCF file of FFPE variants
* A BAM file of the FFPE tumour

#### Command 
The available options are:
* `-o`/`--outdir` - the output directory (default: current directory)
* `-p`/`--prefix` - the output prefix (default: basename of the BAM)

FFPolish can be run with the following command and outputs a filtered vcf `out_filtered.vcf`:
```
ffpolish filter -o outfolder -p out reference.fa vcf.gz tumour.bam
```

### Retraining Model With New Data
We recommend that if you have ground truth data (paired FFPE and fresh frozen tumours), you should create your own dataset to augment the included training set at least with a partial subset of your data.

#### Input Requirements
* A reference genome in FASTA format
* A bgzipped VCF file of FFPE variants
* A BAM file of the FFPE tumour
* A tab-delimited file of true variants
* Output directory

Tab delimited format:
| Column |    Definition    |
|:------:|:----------------:|
| chr    | Chromosome       |
| start  | Start position   |
| end    | End position     |
| ref    | Reference allele |
| alt    | Alternate allele |

#### Command
The available options are:
* `-p`/`--prefix` - the output prefix (default: basename of the BAM)

```
ffpolish extract -p out reference.fa vcf.gz tumour.bam labels.tsv outdir
```
