#!/usr/bin/env python
import argparse
from fastai.vision import *
import time
import geojson


def main(input_file: str, model_path: str, tiling_dir: str, input_tiff_path: str, tiff_geometries_path: str,
         location_name: str, output_file: str):
    job_id = os.path.basename(output_file).split('.json')[0]
    work_dir = os.path.dirname(output_file)

    t1 = time.time()
    # pkl file path
    learn_infer = load_learner(path=os.path.dirname(model_path), file=os.path.basename(model_path))

    tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
    np.random.seed(42)

    # tiling dir path
    test_data = ImageList.from_folder(tiling_dir).split_none().label_empty().transform(
        tfms, size=224, tfm_y=False).databunch().normalize(imagenet_stats)
    test_data.train_dl.new(shuffle=False)
    val_dict = {1: 'yes', 0: 'no'}
    geoms = []

    # input tiff file
    sent_indx = str(input_tiff_path.split('.')[0][-3:])
    with open(input_file, 'r') as f:
        file_data = json.load(f)
    ls_names = file_data['tiling_files']
    print(f"{time.time() - t1}s setup time")

    # unique name for progress csv
    log_path = os.path.join(work_dir, f'{location_name}_{sent_indx}_Inference_Progress_{job_id}.log')
    with open(log_path, 'w+') as f:
        f.write(f'Length of tiling files for job {len(ls_names)}\n')

    # Read in tiff bounding boxes
    with open(tiff_geometries_path, 'r') as f:
        tiff_geometries = geojson.load(f)['geoms']

    # tiling file paths
    for i, tiff_path in enumerate(ls_names):
        t0 = time.time()
        diff = None

        tiff_name = os.path.basename(tiff_path)
        try:
            t1 = time.time()
            im = test_data.train_ds[i][0]
            prediction = learn_infer.predict(im)

            pred_val = prediction[1].data.item()
            pred_label = val_dict[pred_val]
            fl_val = prediction[2].data[pred_val].item()

            geom_bounds = tiff_geometries[tiff_name]
            geoms.append((tiff_path, geom_bounds, pred_label, pred_val, fl_val))
            with open(output_file, 'w+') as f:
                geojson.dump({'geoms': geoms}, f, indent=1)

            outline = f'{i},{tiff_path},{geom_bounds},{pred_label},{pred_val},{fl_val}'
            diff = time.time() - t1

            del im, prediction, pred_val, pred_label, fl_val
        except Exception as e:
            outline = str(e)

        if diff is not None:
            outline += f" inference time: {diff}s"
        outline += f' Total time: {time.time() - t0}s'
        with open(log_path, 'a') as f:
            outline += '\n'
            f.write(outline)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Path to json file with tiling file paths')
    parser.add_argument('--model_path', type=str, required=True, help='Path to ML model')
    parser.add_argument('--tiling_dir', type=str, required=True, help='Path to tiling directory')
    parser.add_argument('--input_tiff', type=str, required=True, help='Path to input tiff file')
    parser.add_argument('--geom_path', type=str, required=True, help='Path to the tiff geometries file')
    parser.add_argument('--location_name', type=str, required=True, help='Location name of job')
    parser.add_argument('--outpath', type=str, required=True, help='Path to the output json file')
    args = parser.parse_args()
    main(args.input_file, args.model_path, args.tiling_dir, args.geom_path, args.input_tiff, args.location_name,
         args.outpath)
