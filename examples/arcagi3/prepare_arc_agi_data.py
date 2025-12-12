import argparse

from rllm.data.dataset import DatasetRegistry

DEFAULT_GAME_IDS = (
    "as66-821a4dcad9c2",
    "ls20-fa137e247ce6",
    "ft09-b8377d4b7815",
    "vc33-6ae7bf49eea5",
    "lp85-d265526edbaa",
    "sp80-0605ab9e5b2a",
)
DEFAULT_ROOT_URL = "https://three.arcprize.org"


def prepare_arc_agi_data(
    dataset_name="arcagi3",
    train_game_ids=None,
    train_repetitions=None,
    test_game_ids=None,
    test_repetitions=None,
    root_url=DEFAULT_ROOT_URL,
    tags=None,
):
    """Register ARC-AGI-3 train/test splits using simple game/repetition lists."""
    print(f"train_game_ids: {train_game_ids}, train_repetitions: {train_repetitions}")
    print(f"test_game_ids: {test_game_ids}, test_repetitions: {test_repetitions}")

    normalized_root = (root_url or "").rstrip("/")
    tag_list = list(tags) if tags else None

    def _build_split(split_name, game_ids, repetitions):
        if not game_ids:
            return []
        if repetitions is None:
            raise ValueError(f"Provide repetitions for the {split_name} split.")
        if len(game_ids) != len(repetitions):
            raise ValueError(
                f"{split_name} split expected {len(game_ids)} repetition counts, got {len(repetitions)} instead."
            )

        data = []
        counter = 0
        for game_id, repeat in zip(game_ids, repetitions):
            if repeat <= 0:
                raise ValueError("Repetition counts must be positive integers.")
            for rollout in range(repeat):
                entry = {
                    "uid": f"{split_name}-{counter}-{game_id}",
                    "game_id": game_id,
                    "root_url": normalized_root,
                    "rollout": rollout,
                }
                if tag_list:
                    entry["tags"] = list(tag_list)
                data.append(entry)
                counter += 1
        return data

    train_ids = train_game_ids or list(DEFAULT_GAME_IDS)
    test_ids = test_game_ids or []

    train_data = _build_split("train", train_ids, train_repetitions)
    test_data = _build_split("test", test_ids, test_repetitions)

    train_dataset = DatasetRegistry.register_dataset(dataset_name, train_data, "train") if train_data else None
    test_dataset = DatasetRegistry.register_dataset(dataset_name, test_data, "test") if test_data else None

    return train_dataset, test_dataset


def _parse_args():
    parser = argparse.ArgumentParser(description="Register ARC-AGI-3 datasets in the DatasetRegistry.")
    parser.add_argument("--dataset-name", default="arcagi3", help="DatasetRegistry name.")
    parser.add_argument(
        "--train-game-ids",
        nargs="*",
        default=None,
        help="Space separated game IDs for training (defaults to all known).",
    )
    parser.add_argument(
        "--train-repetitions",
        nargs="*",
        type=int,
        default=None,
        help="Repetition count per training game.",
    )
    parser.add_argument(
        "--test-game-ids",
        nargs="*",
        default=None,
        help="Space separated game IDs for testing.",
    )
    parser.add_argument(
        "--test-repetitions",
        nargs="*",
        type=int,
        default=None,
        help="Repetition count per testing game.",
    )
    parser.add_argument("--root-url", default=DEFAULT_ROOT_URL, help="ARC backend root URL.")
    parser.add_argument("--tags", nargs="*", default=None, help="Optional tags to attach to every row.")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_dataset, test_dataset = prepare_arc_agi_data(
        dataset_name=args.dataset_name,
        train_game_ids=args.train_game_ids,
        train_repetitions=args.train_repetitions,
        test_game_ids=args.test_game_ids,
        test_repetitions=args.test_repetitions,
        root_url=args.root_url,
        tags=args.tags,
    )

    if train_dataset is not None:
        print(f"Train dataset '{args.dataset_name}' split 'train': {len(train_dataset)} examples")
        print("Sample train example:", train_dataset.get_data()[0])
        print("Sample train example:", train_dataset.get_data()[-1])
    else:
        print("Train dataset was not created (no training games assigned).")

    if test_dataset is not None:
        print(f"Test dataset '{args.dataset_name}' split 'test': {len(test_dataset)} examples")
        print("Sample test example:", test_dataset.get_data()[0])
    else:
        print("Test dataset was not created (no testing games assigned).")


