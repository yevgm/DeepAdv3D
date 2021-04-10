import os
import requests
import tarfile


# support functions
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def check_FAUST2000_dataset(data_dir):
    print('Downloading FAUST dataset')
    if not os.path.isdir(data_dir) or len(os.listdir(data_dir)) == 0:
        os.makedirs(data_dir, exist_ok=True)
        download_file_from_google_drive('1rcNzxuoQ2sfvg2Yv2ffHLgXbOakUwciG', data_dir + '/dataset.tar')
        tf = tarfile.open(data_dir + '/dataset.tar')
        tf.extractall(data_dir)

        os.system('rm %s' % (data_dir + '/dataset.tar'))

def check_shrec14_dataset(data_dir):
    print('Downloading SHREC14 dataset')
    if not os.path.isdir(data_dir) or len(os.listdir(data_dir)) == 0:
        os.makedirs(data_dir, exist_ok=True)
        download_file_from_google_drive('1uuTs-BqxYCJhPtEKjRZofm0c0pnfl3U2', data_dir + '/dataset.tar')
        tf = tarfile.open(data_dir + '/dataset.tar')
        tf.extractall(data_dir)

        os.system('rm %s' % (data_dir + '/dataset.tar'))


if __name__ == '__main__':
    REPO_ROOT = os.path.abspath('.')
    check_FAUST2000_dataset(os.path.abspath(os.path.join(REPO_ROOT, 'datasets/faust')))
    check_shrec14_dataset(os.path.abspath(os.path.join(REPO_ROOT, 'datasets/shrec14')))