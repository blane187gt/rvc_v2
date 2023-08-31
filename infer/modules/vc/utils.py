import os

from fairseq import checkpoint_utils

### don't modify the code before you test it
# def get_index_path_from_model(sid):
#     return next(
#         (
#             f
#             for f in [
#                 os.path.join(root, name)
#                 for root, dirs, files in os.walk(os.getenv("index_root"), topdown=False)
#                 for name in files
#                 if name.endswith(".index") and "trained" not in name
#             ]
#             if sid.split(".")[0] in f
#         ),
#         "",
#     )


def get_index_path_from_model(sid):
    sel_index_path = ""
    name = os.path.join("logs", sid.split(".")[0], "")
    # print(name)
    for f in sel_index_path:
        if name in f:
            # print("selected index path:", f)
            sel_index_path = f
            break
    return sel_index_path


def load_hubert(config):
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["assets/hubert/hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()
