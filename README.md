This fork of JoeyNMT was created as a term project by Christoph Schneider.<br />
It implements Convolutional Sequence to Sequence Learning.<br />
To recreate the work described in the accompanying term paper, please install this version of joeynmt as described here:<br />
	https://joeynmt.readthedocs.io/en/latest/install.html <br />
Please make sure to clone this GitHub repository instead of the original JoeyNMT repository. <br />
To obtain the relevant training data, run the script scripts/getiwslt14_bpe.sh <br />
Then follow the training instructions described here: <br />
	https://joeynmt.readthedocs.io/en/latest/tutorial.html#training <br />
The config file for normal ConvSeq2Seq is called convSeq2Seq_iwslt14_bpe.yaml.  <br />
The config file for ConvSeq2Seq with Multi Head attention is called convSeq2Seq_iwslt14_multihead_bpe.yaml.  <br />
Please note that so far only GPU training is implemented. <br />


