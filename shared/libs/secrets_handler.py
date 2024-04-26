import os, sys, json, shutil, getpass, atexit, time, hashlib

this_dir = libs_dir = os.path.dirname(os.path.abspath(__file__))
shared_dir_path = os.path.dirname(libs_dir)
workspace_dir_path = os.path.join(shared_dir_path, "workspace")
tmp_dir_path = os.path.join(workspace_dir_path, "tmp")
secrets_file_path = os.path.join(tmp_dir_path, "secrets.json")

def init_secrets():
    if not os.path.exists(tmp_dir_path):
        os.makedirs(tmp_dir_path)
    secrets_needed = ["HF_TOKEN",]
    og_secrets_data = {}
    if os.path.exists(secrets_file_path):
        try:
            og_secrets_data = json.load(open(secrets_file_path, 'r'))
        except:
            print("Corrupted secrets file, recreating it.")
    secrets_data = og_secrets_data.copy()
    for secret in secrets_needed:
        if secret not in secrets_data:
            secrets_data[secret] = getpass.getpass(f"{secret}: ")
    if og_secrets_data != secrets_data:
        json.dump(secrets_data, open(secrets_file_path, 'w'))
    return True

def get_secrets():
    return json.load(open(secrets_file_path, 'r'))