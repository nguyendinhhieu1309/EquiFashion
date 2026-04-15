CUDA_VISIBLE_DEVICES=1 python clip_score.py /data/lh/docker/project/TexFit/outputs/texfit_offcial_A2 /data/lh/docker/project/hierafashdiff/evaluations/text_foleder/lle_a2

python evaluations/compute_score.py --real_dir path/to/real --fake_dir path/to/fake --device cuda