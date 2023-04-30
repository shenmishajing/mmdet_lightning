import argparse
import copy
import json
import os
import shutil
from random import sample


def parse_args():
    parser = argparse.ArgumentParser(description="create tiny coco dataset")
    parser.add_argument(
        "--data-path",
        type=str,
        help="path of coco dataset, default data/coco",
        default='data/coco',
    )
    parser.add_argument(
        "--fraction",
        type=int,
        help="fraction of train set to generate, default 1/8",
        default=8,
    )
    parser.add_argument(
        "--split",
        type=str,
        nargs="+",
        help="split of dataset to generate, person_keypoints must be the last one, default all",
        default=["instances", "captions", "person_keypoints"],
    )
    parser.add_argument(
        "--not-copy-test",
        action="store_true",
        help="do not copy test annotations, default copy",
    )
    return parser.parse_args()


def calculate_bbox_distribution(ann):
    res = [0 for _ in ann["categories"]]
    category_map = {c["id"]: i for i, c in enumerate(ann["categories"])}
    for a in ann["annotations"]:
        res[category_map[a["category_id"]]] += 1
    return res


def create_tiny_coco(ann, sample_inds):
    ann = copy.deepcopy(ann)
    ann["images"] = [ann["images"][i] for i in sample_inds]
    image_ids = {img["id"]: i for i, img in enumerate(ann["images"])}
    ann["annotations"] = [a for a in ann["annotations"] if a["image_id"] in image_ids]
    return ann


def main():
    args = parse_args()

    ann_path = os.path.join(args.data_path, "annotations")
    output_path = os.path.join(args.data_path, f"annotations_1_{args.fraction}")
    os.makedirs(output_path, exist_ok=True)

    sample_inds = None
    for name in args.split:
        print(f"loading {name} dataset")
        train_ann = json.load(
            open(os.path.join(ann_path, f"{name}_train2017.json"), "r")
        )

        print(f"sample {name} dataset")
        if sample_inds is None:
            sample_inds = sample(
                range(len(train_ann["images"])),
                len(train_ann["images"]) // args.fraction,
            )
        if name == "person_keypoints":
            sample_inds = {a["image_id"] for a in train_ann["annotations"]}
            sample_inds = [
                i
                for i, img in enumerate(train_ann["images"])
                if img["id"] in sample_inds
            ]
            sample_inds = sample(sample_inds, len(sample_inds) // args.fraction)

        print(f"creating tiny {name} dataset")
        tiny_ann = create_tiny_coco(train_ann, sample_inds)

        print(f"saving tiny {name} dataset")
        json.dump(
            tiny_ann, open(os.path.join(output_path, f"{name}_train2017.json"), "w")
        )

        print(f"copying {name} val annotations")
        shutil.copy2(os.path.join(ann_path, f"{name}_val2017.json"), output_path)

        if name == "instances":
            print("calculating bbox distribution")
            origin_bbox_distribution = [
                x / args.fraction for x in calculate_bbox_distribution(train_ann)
            ]
            tiny_bbox_distribution = calculate_bbox_distribution(tiny_ann)

            print("origin bbox distribution:", origin_bbox_distribution)
            print("tiny bbox distribution:", tiny_bbox_distribution)
            distribution_diff = [
                t - o for t, o in zip(tiny_bbox_distribution, origin_bbox_distribution)
            ]
            print(
                "tiny bbox distribution - origin bbox distribution:", distribution_diff
            )
            distribution_diff = [abs(x) for x in distribution_diff]
            print(
                "max abs of tiny bbox distribution - origin bbox distribution:",
                max(distribution_diff),
            )
            print(
                "min abs of tiny bbox distribution - origin bbox distribution:",
                min(distribution_diff),
            )
            distribution_rate = [
                t / o for t, o in zip(tiny_bbox_distribution, origin_bbox_distribution)
            ]
            print(
                "tiny bbox distribution / origin bbox distribution:", distribution_rate
            )
            print(
                "max tiny bbox distribution / origin bbox distribution:",
                max(distribution_rate),
            )
            print(
                "min tiny bbox distribution / origin bbox distribution:",
                min(distribution_rate),
            )

    # Copy the test annotations
    if not args.not_copy_test:
        print("copying test annotations")
        shutil.copy2(os.path.join(ann_path, "image_info_test2017.json"), output_path)
        shutil.copy2(
            os.path.join(ann_path, "image_info_test-dev2017.json"), output_path
        )


if __name__ == "__main__":
    main()
