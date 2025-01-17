import time
import pickle
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from weiner import WeinerDenoiseFilter
from scipy.stats import multivariate_normal
from skimage.util import view_as_windows as viewW

𝜋 = np.pi


def images_example(path='train_images.pickle'):
    """
    A function demonstrating how to access to image data supplied in this exercise.
    :param path: The path to the pickle file.
    """
    patch_size = (8, 8)

    with open('train_images.pickle', 'rb') as f:
        train_pictures = pickle.load(f)

    patches = sample_patches(train_pictures, psize=patch_size, n=20000)

    plt.figure()
    plt.imshow(train_pictures[0])
    plt.title("Picture Example")

    plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(patches[:, i].reshape(patch_size), cmap='gray')
        plt.title("Patch Example")
    plt.show()


def im2col(A, window, stepsize=1):
    """
    an im2col function, transferring an image to patches of size window (length
    2 list). the step size is the stride of the sliding window.
    :param A: The original image (NxM size matrix of pixel values).
    :param window: Length 2 list of 2D window size.
    :param stepsize: The step size for choosing patches (default is 1).
    :return: A (heightXwidth)x(NxM) matrix of image patches.
    """
    return viewW(np.ascontiguousarray(A), (window[0], window[1])).reshape(-1,
                                                                          window[0] * window[1]).T[:, ::stepsize]


def grayscale_and_standardize(images, remove_mean=True):
    """
    The function receives a list of RGB images and returns the images after
    grayscale, centering (mean 0) and scaling (between -0.5 and 0.5).
    :param images: A list of images before standardisation.
    :param remove_mean: Whether or not to remove the mean (default is True).
    :return: A list of images after standardisation.
    """
    standard_images = []

    for image in images:
        standard_images.append((0.299 * image[:, :, 0] +
                                0.587 * image[:, :, 1] +
                                0.114 * image[:, :, 2]) / 255)

    sum = 0
    pixels = 0
    for image in standard_images:
        sum += np.sum(image)
        pixels += image.shape[0] * image.shape[1]
    dataset_mean_pixel = float(sum) / pixels

    if remove_mean:
        for image in standard_images:
            image -= np.matlib.repmat([dataset_mean_pixel], image.shape[0],
                                      image.shape[1])

    return standard_images


def sample_patches(images, psize=(8, 8), n=10000, remove_mean=True):
    """
    sample N p-sized patches from images after standardising them.

    :param images: a list of pictures (not standardised).
    :param psize: a tuple containing the size of the patches (default is 8x8).
    :param n: number of patches (default is 10000).
    :param remove_mean: whether the mean should be removed (default is True).
    :return: A matrix of n patches from the given images.
    """
    d = psize[0] * psize[1]
    patches = np.zeros((d, n))
    standardized = grayscale_and_standardize(images, remove_mean)

    shapes = []
    for pic in standardized:
        shapes.append(pic.shape)

    rand_pic_num = np.random.randint(0, len(standardized), n)
    rand_x = np.random.rand(n)
    rand_y = np.random.rand(n)

    for i in range(n):
        pic_id = rand_pic_num[i]
        pic_shape = shapes[pic_id]
        x = int(np.ceil(rand_x[i] * (pic_shape[0] - psize[1])))
        y = int(np.ceil(rand_y[i] * (pic_shape[1] - psize[0])))
        patches[:, i] = np.reshape(np.ascontiguousarray(
            standardized[pic_id][x:x + psize[0], y:y + psize[1]]), d)

    return patches


def denoise_image(Y, model, denoise_function, noise_std, patch_size=(8, 8)):
    """
    A function for denoising an image. The function accepts a noisy gray scale
    image, denoises the different patches of it and then reconstructs the image.

    :param Y: the noisy image.
    :param model: a Model object (MVN/ICA/GSM).
    :param denoise_function: a pointer to one of the denoising functions (that corresponds to the model).
    :param noise_std: the noise standard deviation parameter.
    :param patch_size: the size of the patch that the model was trained on (default is 8x8).
    :return: the denoised image, after each patch was denoised. Note, the denoised image is a bit
    smaller than the original one, since we lose the edges when we look at all of the patches
    (this happens during the im2col function).
    """
    (h, w) = np.shape(Y)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))

    # split the image into columns and denoise the columns:
    noisy_patches = im2col(Y, patch_size)
    denoised_patches = denoise_function(noisy_patches, model, noise_std)

    # reshape the denoised columns into a picture:
    x_hat = np.reshape(denoised_patches[middle_linear_index, :],
                       [cropped_h, cropped_w])

    return x_hat


def crop_image(X, patch_size=(8, 8)):
    """
    crop the original image to fit the size of the denoised image.
    :param X: The original picture.
    :param patch_size: The patch size used in the model, to know how much we need to crop.
    :return: The cropped image.
    """
    (h, w) = np.shape(X)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))
    columns = im2col(X, patch_size)
    return np.reshape(columns[middle_linear_index, :], [cropped_h, cropped_w])


def normalize_log_likelihoods(X):
    """
    Given a matrix in log space, return the matrix with normalized columns in
    log space.
    :param X: Matrix in log space to be normalised.
    :return: The matrix after normalization.
    """
    h, w = np.shape(X)
    return X - np.matlib.repmat(logsumexp(X, axis=0), h, 1)


def test_denoising(image, model, denoise_function,
                   noise_range=(0.01, 0.05, 0.1, 0.2), patch_size=(8, 8)):
    """
    A simple function for testing your denoising code. You can and should
    implement additional tests for your code.
    :param image: An image matrix.
    :param model: A trained model (MVN/ICA/GSM).
    :param denoise_function: The denoise function that corresponds to your model.
    :param noise_range: A tuple containing different noise parameters you wish
            to test your code on. default is (0.01, 0.05, 0.1, 0.2).
    :param patch_size: The size of the patches you've used in your model.
            Default is (8, 8).
    """
    h, w = np.shape(image)
    noisy_images = np.zeros((h, w, len(noise_range)))
    denoised_images = []
    cropped_original = crop_image(image, patch_size)

    # make the image noisy:
    for i in range(len(noise_range)):
        noisy_images[:, :, i] = image + (
                noise_range[i] * np.random.randn(h, w))

    # denoise the image:
    for i in range(len(noise_range)):
        denoised_images.append(
            denoise_image(noisy_images[:, :, i], model, denoise_function,
                          noise_range[i], patch_size))

    # calculate the MSE for each noise range:
    for i in range(len(noise_range)):
        print("noisy MSE for noise = " + str(noise_range[i]) + ":")
        print(np.mean((crop_image(noisy_images[:, :, i],
                                  patch_size) - cropped_original) ** 2))
        print("denoised MSE for noise = " + str(noise_range[i]) + ":")
        print(np.mean((cropped_original - denoised_images[i]) ** 2))

    plt.figure()
    plt.suptitle(model.__class__.__name__)
    for i in range(len(noise_range)):
        plt.subplot(2, len(noise_range), i + 1)
        plt.imshow(noisy_images[:, :, i], cmap='gray')
        plt.subplot(2, len(noise_range), i + 1 + len(noise_range))
        plt.imshow(denoised_images[i], cmap='gray')
    plt.show()


class MVN_Model:
    """
    A class that represents a Multivariate Gaussian Model, with all the parameters
    needed to specify the model.

    mean - a D sized vector with the mean of the gaussian.
    cov - a D-by-D matrix with the covariance matrix.
    """

    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov


class GSM_Model:
    """
    A class that represents a GSM Model, with all the parameters needed to specify
    the model.

    cov - a k-by-D-by-D tensor with the k different covariance matrices. the
        covariance matrices should be scaled versions of each other.
    mix - k-length probability vector for the mixture of the gaussians.
    """

    def __init__(self, cov, mix):
        self.cov = cov
        self.mix = mix


class ICA_Model:
    """
    A class that represents an ICA Model, with all the parameters needed to specify
    the model.

    P - linear transformation of the sources. (X = P*S)
    vars - DxK matrix whose (d,k) element corresponds to the variance of the k'th
        gaussian in the d'th source.
    mix - DxK matrix whose (d,k) element corresponds to the weight of the k'th
        gaussian in d'th source.
    """

    def __init__(self, P, vars, mix):
        self.P = P
        self.vars = vars
        self.mix = mix


def MVN_log_likelihood(X, model):
    """
    Given image patches and a MVN model, return the log likelihood of the patches
    according to the model.

    :type model: MVN_Model
    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A MVN_Model object.
    :return: The log likelihood of all the patches combined.
    """
    return np.sum(multivariate_normal.logpdf(X.T, model.mean, model.cov))


def GSM_log_likelihood(X, model):
    """
    Given image patches and a GSM model, return the log likelihood of the patches
    according to the model.

    :type model: GSM_Model
    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A GSM_Model object.
    :return: The log likelihood of all the patches combined.
    """

    D, N = X.shape
    k = len(model.mix)
    mean = np.zeros(D) # mean is always 0
    # ll = np.zeros(N)
    # for i in range(N):
    #     for y in range(k):
    #         ll[i] += np.log(model.mix[y]) * multivariate_normal.pdf(X[:, i], mean, model.cov[y])

    pdf_mat = np.zeros((k, N))
    for y in range(k):
        pdf_mat[y] = multivariate_normal.pdf(X.T, mean, model.cov[y]) * model.mix[y]

    return np.sum(np.log(np.sum(pdf_mat, axis=0)))


def ICA_log_likelihood(X, model):
    """
    Given image patches and an ICA model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: An ICA_Model object.
    :type model: ICA_Model
    :return: The log likelihood of all the patches combined.
    """

    # TODO: YOUR CODE HERE


def learn_MVN(X):
    """
    Learn a multivariate normal model, given a matrix of image patches.
    :rtype: MVN_Model
    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :return: A trained MVN_Model object.
    """
    mean = np.mean(X, 1)
    cov = np.cov(X)
    return MVN_Model(mean, cov)


def learn_GSM(X, k):
    """
    Learn parameters for a Gaussian Scaling Mixture model for X using EM.

    GSM components share the variance, up to a scaling factor, so we only
    need to learn scaling factors and mixture proportions.

    :rtype: GSM_Model
    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components of the GSM model.
    :return: A trained GSM_Model object.
    """

    MAX_ITERS = 120
    EPSILON = 1e-2


    # Init variables
    D, N = np.shape(X)
    random = np.random.rand(k)
    𝜋 = random / np.sum(random)
    mean = np.zeros((D))  # Mean is always 0
    cov = np.cov(X)  # Base covariance is sample covarivance
    inv_cov = np.linalg.inv(cov)
    r_squared = np.random.rand(k)
    covs = np.zeros((k, D, D))
    for y in range(k):
        covs[y] = cov * r_squared[y]
    c = np.zeros((k, N))

    prev_ll = -EPSILON - 50
    ll_stats = []
    iteration = 1
    # EM
    while iteration < MAX_ITERS:
        # E-step
        log_pdfs = np.zeros((k, N))
        for y in range(k):
            log_pdfs[y] = multivariate_normal.logpdf(X.T, mean, covs[y])

        log_pdfs = (log_pdfs.T + np.log(𝜋)).T # Adding the log of 𝜋_y to each corresponding row
        this_ll = GSM_log_likelihood(X, GSM_Model(covs, 𝜋))
        c = normalize_log_likelihoods(log_pdfs)

        print(f'Finished  E-step {iteration}')

        if np.abs(this_ll - prev_ll) < EPSILON:
            break

        # M-step
        c_sum = np.sum(np.exp(c), axis=1)
        𝜋 = c_sum / N # updating probabilities

        # Updating r_squared and covariance mats
        for y in range(k):
            new_r = 0
            for i in range(N):
                new_r += np.exp(c[y, i]) * (X[:, i].T @ inv_cov @ X[:, i])
            r_squared[y] = new_r / (D * c_sum[y])
            covs[y] = cov * r_squared[y]
        print(f'Finished M-step {iteration}')

        print(f'Done iteration {iteration}')
        print(f'LL for this iteration: {this_ll}')
        ll_stats.append(this_ll)

        with open(f'gsm/it_{iteration}', 'wb') as f:
            pickle.dump({'covs': covs, 'rs': r_squared, 'pi': 𝜋, 'c': c, 'LL' : ll_stats}, f)

        prev_ll = this_ll
        iteration += 1
        

    plt.title('Log-likelihood as a function of iteration number')
    plt.plot(ll_stats)
    plt.show()

    return GSM_Model(covs, 𝜋)


def learn_ICA(X, k):
    """
    Learn parameters for a complete invertible ICA model.

    We learn a matrix P such that X = P*S, where S are D independent sources
    And for each of the D coordinates we learn a mixture of K univariate
    0-mean gaussians using EM.

    :rtype: ICA_Model
    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components in the source gaussian mixtures.
    :return: A trained ICA_Model object.
    """

    # TODO: YOUR CODE HERE


def MVN_Denoise(Y, mvn_model, noise_std):
    """
    Denoise every column in Y, assuming an MVN model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by a single
    0-mean multi-variate normal distribution.

    :type mvn_model: MVN_Model
    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param mvn_model: The MVN_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """
    start = time.time()
    mean = np.zeros((mvn_model.cov.shape[0]))  # Since we know mean is 0
    filter = WeinerDenoiseFilter(mean, mvn_model.cov, noise_std)
    result = np.apply_along_axis(filter, 0, Y)
    print(f'MVN Filtered Image in {time.time() - start:.2f} seconds')
    return result


def GSM_Denoise(Y, gsm_model, noise_std):
    """
    Denoise every column in Y, assuming a GSM model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by a mixture of
    0-mean gaussian components sharing the same covariance up to a scaling factor.

    :type gsm_model: GSM_Model
    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param gsm_model: The GSM_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.

    """
    start = time.time()

    D, M = Y.shape
    k = len(gsm_model.mix)
    result = np.zeros((D, M))
    mean = np.zeros(D)
    filters = [WeinerDenoiseFilter(mean, cov, noise_std) for cov in gsm_model.cov]
    noise_covs = [gsm_model.cov[i] + np.identity(D) * noise_std**2 for i in range(k)]

    all_c = np.zeros((k, M))
    for i in range(k):
        all_c[i] = np.log(gsm_model.mix[i]) + multivariate_normal.logpdf(Y.T, mean, noise_covs[i])
    all_c = all_c / all_c.sum(axis=0)

    for col in range(Y.shape[1]):
        patch = Y[:, col]
        c = all_c[:, col]
        filterd = np.array([filters[i](patch) for i in range(k)])
        result[:, col] = np.dot(filterd.T, c)

    print(f'GSM Filtered Image in {time.time() - start:.2f} seconds')
    return result


def ICA_Denoise(Y, ica_model, noise_std):
    """
    Denoise every column in Y, assuming an ICA model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by an ICA 0-mean
    mixture model.

    :type ica_model: ICA_Model
    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param ica_model: The ICA_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """

    # TODO: YOUR CODE HERE


if __name__ == '__main__':
    patch_size = (8, 8)

    with open('train_images.pickle', 'rb') as f:
        train_pictures = pickle.load(f)

    with open('test_images.pickle', 'rb') as f:
        test_pictures = pickle.load(f)

    train_patches = sample_patches(train_pictures, psize=patch_size, n=20000)

    with open('gsm3/it_35', 'rb') as f:
        s = pickle.load(f)

    #
    # ll = s['LL']
    # plt.plot(ll)
    # plt.show()

    # model_gsm = GSM_Model(s['covs'], s['pi'])
    start = time.time()
    model_mvn = learn_MVN(train_patches)
    print(f'MVN learned in {time.time() - start} secs')

    start = time.time()
    model_gsm = learn_GSM(train_patches, 6)
    print(f'GSM learned in {time.time() - start} secs')

    print('Done training')
    print(f'MVN log-likelihood on clean images is: {MVN_log_likelihood(train_patches, model_mvn):.3f}')
    print(f'GSM log-likelihood on clean images is: {GSM_log_likelihood(train_patches, model_gsm):.3f}')
    #
    std_test_pics = grayscale_and_standardize(test_pictures)
    # Low noise tests
    test_denoising(std_test_pics[2], model_mvn, MVN_Denoise, patch_size=patch_size, noise_range=(0.01, 0.1, 0.3))
    test_denoising(std_test_pics[2], model_gsm, GSM_Denoise, patch_size=patch_size, noise_range=(0.01, 0.1, 0.3))

    # High Noise tests
    test_denoising(std_test_pics[2], model_mvn, MVN_Denoise, patch_size=patch_size, noise_range=(0.7, 0.9, 1))
    test_denoising(std_test_pics[2], model_gsm, GSM_Denoise, patch_size=patch_size, noise_range=(0.7, 0.9, 1))
