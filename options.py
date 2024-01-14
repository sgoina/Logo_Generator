import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    # basic parameters: data shapes
    parser.add_argument('--input_size', type=int, default=64, help='the size of glyph image (character or word)')
    parser.add_argument('--output_size', type=int, default=512, help='the size of logo image')
    parser.add_argument('--max_seqlen', type=int, default=20, help='the max length of charcters(words)')
    parser.add_argument('--in_channel', type=int, default=3, help='the input glyph image channel')
    parser.add_argument('--output_channel', type=int, default=3, help='the logo image channel')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--latent_dim', type=int, default=128, help='batch size')
    parser.add_argument('--embed_dim', type=int, default=300, help='the dim of char/word embeddings')
    parser.add_argument('--pos_format', type=str, default='ltrb', choices=['whxy', 'ltrb'], help='the format of element position, width-height-centre_x-centre_y or left-top-right-bottom')
    # conditon encoding related
    parser.add_argument('--cond_modality', type=str, default='img_text', choices=['img_text', 'img', 'text'], help='whether to use word embeddings')
    # parser.add_argument('--embeds_corpus', type=str, default='baidubaike', choices=['baidubaike', 'wiki'], help='which corpus of text embeds to use')
    parser.add_argument('--use_embed_word', type=bool, default=True, help='whether to use word embeddings')
    parser.add_argument('--glyph_rec', type=bool, default=False, help='use glyph reconsturction as additional supervision')
    parser.add_argument('--loss_pt_c_w', type=float, default=0.001, help='the weight of perceptual content loss')
    parser.add_argument('--loss_rec_l1_w', type=float, default=0.1, help='the loss weight of rec l1 loss')
    # Sequence model (LSTM) realted
    parser.add_argument('--hidden_size', type=int, default=128, help='LSTM hidden size')
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='LSTM number of hidden layer')
    # loss weight
    parser.add_argument('--loss_ol_w', type=float, default=100.0, help='the loss weight of overlap loss')
    # discriminator related
    parser.add_argument('--n_train_imgdis_interval', type=int, default=5, help='the interval of training Image Discriminator')
    parser.add_argument('--loss_seqdis_w', type=int, default=1.0, help='the loss weight of seqdis')
    parser.add_argument('--loss_imgdis_w', type=int, default=0.01, help='the loss weight of imgdis')
    parser.add_argument('--DG_read_diff_data', type=bool, default=True, help='Discriminator and Generator read different data, to avoid overfitting')
    parser.add_argument('--imgdis_logo_ms', type=bool, default=False, help='whether to use glyphs with different grey scales as img dis input')
    parser.add_argument('--opt_G_jointly', type=bool, default=False, help='whether to optimize the G_seq and G_img loss jointly')
    # Diff Composition Related
    parser.add_argument('--align_corners', type=bool, default=False, help='align_corners in affine 2D')
    parser.add_argument('--trunc_logo_pix', type=bool, default=False, help='whether to trunc the pixel value of composited logo images into [0, 1] ([0, 255]) for discrimination')
    # experiment related
    parser.add_argument('--data_root', type=str, default='dataset')
    parser.add_argument('--data_name', type=str, default='YourDataSet')
    parser.add_argument('--experiment_name', type=str, default='base_model')
    parser.add_argument('--init_epoch', type=int, default=0, help='init epoch')
    parser.add_argument('--multi_gpu', type=bool, default=True, help='whether to use multi-gpu')
    parser.add_argument('--n_epochs', type=int, default=800, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--n_ckpt_interval', type=int, default=20, help='save checkpoint frequency of epoch')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of Adam optimizer')
    parser.add_argument('--eps', type=float, default=1e-8, help='Adam epsilon')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--tboard', type=bool, default=True, help='whether use tensorboard to visulize loss')
    parser.add_argument('--n_summary_interval', type=int, default=50, help='the interval of batches when writing summary')
    parser.add_argument('--n_log_interval', type=int, default=10, help='the interval of batches when printing summary')
    # testing and validation related
    parser.add_argument('--test_epoch', type=int, default=600, help='the epoch for testing')
    parser.add_argument('--test_sample_times', type=int, default=10, help='the number of sampling times for each test case when testing')
    parser.add_argument('--train_sample_times', type=int, default=5, help='the number of sampling times for each test case when training')
    
    # gen_data
    parser.add_argument("--input_text", type=str, default='你好')
    parser.add_argument("--ttf_path", type=str, default='./dataset/ttfs/FZShangKJW.TTF')
    parser.add_argument('--canvas_size', type=int, default=300)
    parser.add_argument('--glyph_size', type=int, default=64)
    parser.add_argument('--logo_size', type=int, default=128)
    parser.add_argument('--starting_pos', type=int, default=20, help='the starting position')
    parser.add_argument("--glyph_output_dir", type=str, default='./dataset/YourDataSet/')
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--embed_path", type=str, default='./dataset/Embeddings/sgns.baidubaike.bigram-char')

    # Specify inputs and outputs
    parser.add_argument('--content', type=str, default='texture_style/YourDataSet/000.png', help="File path to the content image, valid for style transfer and invalid for texture transfer")
    parser.add_argument('--style', type=str, default='texture_style/style/input-sem.png', help="File path to the style/source image")
    parser.add_argument('--content_sem', type=str, default='texture_style/style/input-paint.png', help="File path to the semantic map of content/target image")
    parser.add_argument('--style_sem', type=str, default='inputs/doodles/Gogh_sem.png', help="File path to the semantic map of style/source image")
    parser.add_argument('--outf', type=str, default='outputs', help="Folder to save output images")
    parser.add_argument('--output', type=str, default='output', help="Name of output images")
    parser.add_argument('--content_size', type=int, default=0, help="Resize content/target, leave it to 0 if not resize")
    parser.add_argument('--style_size', type=int, default=0, help="Resize style/source, leave it to 0 if not resize")
    parser.add_argument('--style_transfer', action="store_true", help="Activate it if you want style transfer rather than texture transfer")

    # Runtime controls
    parser.add_argument('--coarse_alpha', type=float, default=1, help="Hyperparameter to blend transformed feature with content feature in coarse level (level 5)")
    parser.add_argument('--fine_alpha', type=float, default=1, help="Hyperparameter to blend transformed feature with content feature in fine level (level 4)")

    parser.add_argument('--semantic', type=str, default='concat', choices=['add', 'concat', 'concat_ds'], help="Choose different modes to embed semantic maps, 'add': our addition, 'concat': our concatenation, 'concat_ds': concat downsampled semantic maps")
    parser.add_argument('--concat_weight', type=float, default=50, help="Hyperparameter to control the semantic guidance/awareness weight for '-semantic concat' mode and '-semantic concat_ds' mode, range 0-inf")
    parser.add_argument('--add_weight', type=float, default=0.6, help="Hyperparameter to control the semantic guidance/awareness weight for '-semantic add' mode, range 0-1")

    parser.add_argument('--coarse_psize', type=int, default=0, help="Patch size in coarse level (level 5), 0 means using global view")
    parser.add_argument('--fine_psize', type=int, default=3, help="Patch size in fine level (level 4)")

    parser.add_argument('--enhance', type=str, default='adain', choices=['adain', 'wct'], help="Choose different enhancement modes in level 3, level 2, and level 1. 'adain': first-order statistics enhancement, 'wct': second-order statistics enhancement.")
    parser.add_argument('--enhance_alpha', type=float, default=1, help="Hyperparameter to control the enhancement degree in level 3, level 2, and level 1")

    # Compress models
    parser.add_argument('--compress', action="store_true", help="Use the compressed models for faster inference")
    return parser