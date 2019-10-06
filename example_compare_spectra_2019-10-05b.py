#!/usr/bin/env python

# Prerequisites:
# 1) Download and compile visfd
#    git clone https://github.com/jewettaij/visfd ~/visfd
#    cd ~/visfd
#    bash   #<-- make sure you are using the BASH shell
#    source alternate_compiler_settings/for_gcc/setup_gcc.sh
#    make clean
#    make
#    mkdir ~/bin
#    export PATH="$PATH:$HOME/bin"  #<-- make sure $HOME/bin is in your PATH !
#    cp bin/crop_mrc ~/bin/
#
# 2) crop a (preferably cubic) region from the image
#
# crop_mrc Stevens_pancreatic_INS_1E_25mM_769_7_pre_rec.mrc orig_crop_98_225_133_260_178_305_DELETEME.mrc 98 225 133 260 178 305
#
#    Do the same thing to the segmented version of the image
#
# crop_mrc Stevens_pancreatic_INS_1E_25mM_769_7_Final_pre_rec.Labels.mrc orig_labels_98_225_133_260_178_305_DELETEME.mrc 98 225 133 260 178 305
#
# orig_labels_98_225_133_260_178_305.mrc
# 
# 2) Use IMOD (specifically IMOD's "newstack" command) to read the segmented
#    file (name ending in ".Labels.mrc") and create a new file that is 
#    MRC/CCP4 2014 compliant:
#
# newstack -mode 2 -in orig_crop_98_225_133_260_178_305_DELETEME.mrc  -ou orig_crop_98_225_133_260_178_305.mrc
# rm -f orig_crop_98_225_133_260_178_305_DELETEME.mrc
#
#    Do the same thing to the segmented version of the image
#
# newstack -mode 2 -in orig_labels_98_225_133_260_178_305_DELETEME.mrc  -ou orig_labels_98_225_133_260_178_305.mrc
# rm -f orig_labels_98_225_133_260_178_305_DELETEME.mrc
#
# 4) Create a version of the previously segmented image with the
#    a) extracellular space (0) replaced by the number 0.0045
#    b) nucleus (1) replaced by 0.009
#    b) ISGs (2) replaced by the number 0.0145
#    c) Organelles (3) replaced by 0.0125
#    e) remaining cytoplasm (4) replaced by 0.0101
#
# import np
# import mrcfile
#
# mp = {0:0.0045, 1:0.009, 2:0.0145, 3:0.0125, 4:0.0101}
#
# img_src = mrcfile.open('orig_labels_98_225_133_260_178_305.mrc')
# img_size = img_src.data.shape
#
# img_dest = mrcfile.new('orig_labels_float_98_225_133_260_178_305.mrc', overwrite=True)
# img_dest.set_data(np.zeros(img_size, dtype=np.float32))
#
## the next loop takes a while
#for iz in range(0, img_size[2]):
#    # print(iz)   # <-- (uncomment this to monitor progress)
#    for iy in range(0, img_size[1]):
#        for ix in range(0, img_size[0]):
#            img_dest.data[iz][iy][ix] = mp[img_src.data[iz][iy][ix]]
# 
# img_dest.close()


from math import *
import numpy as np


def SpectralR3d(img):
    """calculate frequency spectra in a 3d image 
       as a function of the fourier index magnitude k=sqrt(kx^2+ky^2+kz^2).
       If the image is not a cube, then the analysis will be performed on
       a cube-shaped image of size N extracted from the center of the original
       image where N = min(img.shape) (ie the shortest edge of the rectangle).
       This function returns two arrays of the same size:
       1) an array of the average |FT| vs k, where k=sqrt(kx^2+ky^2+kz^2)
           and kx,ky,kz are integers which are the discrete frequencies
           considered by the discrete fourier transform.
       2) an array of spatial frequencies (angular frequencies)
          k = 2πν/N, where ν = 0,1,2,...⌊N/2⌋  (where ⌊N/2⌋ = floor(N/2) = N//2)
    """
    img_size = img.shape
    Nx = img_size[0]
    Ny = img_size[1]
    Nz = img_size[2]
    N = min(img_size)  # = min(Nx, Ny, Nz)
    N_2 = N//2   # = int(floor(N/2))

    # Make sure the image is a cube (Nx = Ny = Nz = N)
    # by discarding (cropping) voxels outside a cube of this size
    img = img[Nx//2 - int(floor(N/2)) : Nx//2 + int(ceil(N/2)),
              Ny//2 - int(floor(N/2)) : Ny//2 + int(ceil(N/2)),
              Nz//2 - int(floor(N/2)) : Nz//2 + int(ceil(N/2))]
    assert(img.shape == (N, N, N))

    # Compute the fourier transformed of the (cropped cubical) image:
    img_ft = np.fft.fftn(img)

    kr_max = N//2+1

    # store radial fourier transform here
    img_ft_vs_kr = np.array([0.0 for kr in range(0, kr_max)])
    # temporary variable:
    n_kr = np.array([0 for kr in range(0, kr_max)])

    # Now average the magnitude of the fourier expansion at constant k
    # where k = sqrt(kx^2 + ky^2 + kz^2)   (using a bin-width of 1 <--> 2π/N)

    # Consider a discrete Fourier series: Σ_k=0^{N-1} a_k exp(i2πkn/N)
    # (Notation: Σ_k=0^{N-1} denotes the sum from k=0...N-1,
    #            and a_k is the kth Fourier coefficient in the expansion)
    # Consider the identity: exp(iθ) = exp(i(θ+2πL)) where L is any integer
    # setting L=-1      -->  exp(iθ) = exp(-i(2π-θ))
    # setting θ=2πk/N   -->  exp(i2πk/N)  = exp(-i2π(N - k)/N)
    # exponentiating both sides of the equation to the power of n yields:
    #             ^n    -->  exp(i2πkn/N) = exp(-i2π(N - k)n/N)
    # This means that the kth term in the Fourier series, exp(i2πkn/N) can be
    # replaced by exp(-i2π(N - k)n/N).  It's convenient to do that when k>N/2,
    # because the magnitude of the spatial frequencies above N/2 are N-k not k
    # (*2πn/N).  And for this analysis, we care about the magnitude of the
    # spatial frequencies in the image, not their sign.
    for kz in range(0, N):
        for ky in range(0, N):
            for kx in range(0, N):
                Kx = min(kx, N-kx)  # see explanation above
                Ky = min(ky, N-ky)  # see explanation above
                Kz = min(kz, N-kz)  # see explanation above
                kr = floor(sqrt(Kx*Kx+Ky*Ky+Kz*Kz))
                if kr >= kr_max:
                    continue
                n_kr[kr] += 1
                img_ft_vs_kr[kr] += abs(img_ft[kz,ky,kx])

    for kr in range(0, kr_max):
        img_ft_vs_kr[kr] /= n_kr[kr]

    frequencies = np.array([2*pi*k/N for k in range(0, kr_max)])

    return img_ft_vs_kr, frequencies



def SpectralRxy(img):
    """calculate frequency spectra in a 3d image 
       as a function of the fourier index magnitude kxy=sqrt(kx^2+ky^2).
       If the image is not a cube, then the analysis will be performed on
       a cube-shaped image of size N extracted from the center of the original
       image where N = min(img.shape) (ie the shortest edge of the rectangle).
       This function returns two arrays of the same size:
       1) an array of the average |FT| vs k, where k=sqrt(kx^2+ky^2+kz^2)
           and kx,ky,kz are integers which are the discrete frequencies
           considered by the discrete fourier transform.
       2) an array of spatial frequencies (angular frequencies)
          k = 2πν/N, where ν = 0,1,2,...⌊N/2⌋  (where ⌊N/2⌋ = floor(N/2) = N//2)
    """
    img_size = img.shape
    Nx = img_size[0]
    Ny = img_size[1]
    Nz = img_size[2]
    N = min(img_size)  # = min(Nx, Ny, Nz)
    N_2 = N//2   # = int(floor(N/2))

    # Make sure the image is a cube (Nx = Ny = Nz = N)
    # by discarding (cropping) voxels outside a cube of this size
    img = img[Nx//2 - int(floor(N/2)) : Nx//2 + int(ceil(N/2)),
              Ny//2 - int(floor(N/2)) : Ny//2 + int(ceil(N/2)),
              Nz//2 - int(floor(N/2)) : Nz//2 + int(ceil(N/2))]
    assert(img.shape == (N, N, N))

    # Compute the fourier transformed of the (cropped cubical) image:
    img_ft = np.fft.fftn(img)

    kr_max = N//2+1

    # store radial fourier transform here
    img_ft_vs_kr = np.array([0.0 for kr in range(0, kr_max)])
    # temporary variable:
    n_kr = np.array([0 for kr in range(0, kr_max)])

    # Now average the magnitude of the fourier expansion at constant k
    # where k = sqrt(kx^2 + ky^2)   (using a bin-width of 1 <--> 2π/N)

    kz = 0
    for ky in range(0, N):
        for kx in range(0, N):
            Kx = min(kx, N-kx)  # see explanation above
            Ky = min(ky, N-ky)  # see explanation above
            Kz = min(kz, N-kz)  # see explanation above
            kr = floor(sqrt(Kx*Kx+Ky*Ky+Kz*Kz))
            if kr >= kr_max:
                continue
            n_kr[kr] += 1
            img_ft_vs_kr[kr] += abs(img_ft[kz,ky,kx])

    for kr in range(0, kr_max):
        img_ft_vs_kr[kr] /= n_kr[kr]

    frequencies = np.array([2*pi*k/N for k in range(0, kr_max)])

    return img_ft_vs_kr, frequencies



# Load the image data from the MRC file
import mrcfile
img_mrc = mrcfile.open('orig_crop_98_225_133_260_178_305.mrc')
img_spectrum, frequencies = SpectralR3d(img_mrc.data)
#img_spectrum, frequencies = SpectralRxy(img_mrc.data)

img_labels_mrc = mrcfile.open('orig_labels_float_98_225_133_260_178_305.mrc')
img_labels_spectrum, frequencies = SpectralR3d(img_labels_mrc.data)
#img_labels_spectrum, frequencies = SpectralRxy(img_labels_mrc.data)

# Plot the frequency spectra
import matplotlib.pyplot as plt

plt.step(frequencies, img_spectrum, where='mid', label='original')
plt.step(frequencies, img_labels_spectrum, where='mid',label='segmented')

# If you display the whole spectrum, all you will see are the largest 1 or 2
# entries in the spectrum (probably at low frequencies).
# So instead of showing the whole spectrum, scale the Y axis so that you can
# see a certain fraction of them ("median_fraction", for example 85% of them).
img_spectrum_sorted = img_spectrum
img_spectrum_sorted.sort()
median_fraction = 0.85
median_fraction_i = int(floor(median_fraction*len(img_spectrum_sorted)))
ymax = img_spectrum_sorted[median_fraction_i]

plt.title('frequency spectra of the image')
plt.ylabel('⟨|FT(image)|⟩')
plt.xlabel('Angular Spatial Frequency (ω=2πν)')
plt.legend(title='images')
plt.ylim(bottom=0, top=ymax)
plt.xlabel('Frequency (ω=2πν)')
plt.ylabel('⟨|FT(image)|⟩')
plt.ylim(bottom=0, top=ymax)
plt.show()


# Finally, compare the two spectra by dividing one by the other:

#plt.step(frequencies, img_spectrum / img_labels_spectrum)
#ymax=2         
#plt.ylim(bottom=0, top=ymax)
#plt.show()



