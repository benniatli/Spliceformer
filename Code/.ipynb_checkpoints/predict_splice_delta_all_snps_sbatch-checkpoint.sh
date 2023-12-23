# +
# #!/bin/bash
#SBATCH --job-name Splice-Variant-Prediction
#SBATCH -p dgxa100-test
#SBATCH --mem 32g
#SBATCH --time 48:00:00
#SBATCH -c 1

# cd /odinn/users/benediktj/splice-site-prediction/Code/
apptainer exec --nv -H /odinn/tmp/benediktj/Singularity/pytorch -B /odinn/data/dataprocessing/:/odinn/data/dataprocessing/ -B /nfs/fs1/bioinfo/:/nfs/fs1/bioinfo/ -B /nfs/odinn/users/gislih/RNA/requests/rna_paper/splice_anno/:/nfs/odinn/users/gislih/RNA/requests/rna_paper/splice_anno/ -B /usr/bin/uname:/usr/bin/uname -B /etc/ssl/certs/ca-bundle.crt/:/etc/ssl/certs/ca-bundle.crt -B /odinn/tmp/benediktj/:/odinn/tmp/benediktj/ -B /odinn/data/reference/Homo_sapiens-deCODE-hg38/Sequence/WholeGenomeFasta/genome.fa:/odinn/data/reference/Homo_sapiens-deCODE-hg38/Sequence/WholeGenomeFasta/genome.fa -B /nfs/prog/bioinfo/apps-x86_64/bin/samtools:/nfs/prog/bioinfo/apps-x86_64/bin/samtools -B /odinn/users/gislih/RNA/requests/rna_paper/splice_anno/:/odinn/users/gislih/RNA/requests/rna_paper/splice_anno/ -B /odinn/users/solvir/requests/magnus_ulfars/:/odinn/users/solvir/requests/magnus_ulfars/ -B /odinn/groups/machinelearning/:/odinn/groups/machinelearning/ -B /odinn/users/benediktj/splice-site-prediction:/splice-site-prediction sh /odinn/users/benediktj/DockerImages/pytorch_21.10-py3.sif $split_nr
