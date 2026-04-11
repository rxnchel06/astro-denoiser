import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.visualization import ZScaleInterval
from astropy.stats import sigma_clip
from skimage.filters import gaussian, median
from skimage.morphology import disk
from skimage.restoration import richardson_lucy, estimate_sigma

st.set_page_config(page_title="🔭 Astro Denoiser", layout="wide")
st.title("🔭 Astronomical Image Denoiser")
st.caption("Real physics-based noise reduction for astronomical images")

# --- Load image ---
# --- Image Source ---
st.sidebar.header("📁 Image Source")
source = st.sidebar.radio("Choose source", [
    "Upload my own FITS file",
    "Use sample images"
])

if source == "Upload my own FITS file":
    uploaded = st.sidebar.file_uploader("Upload a FITS file", type=["fits", "fit"])
    if uploaded is not None:
        with fits.open(uploaded) as hdul:
            data = None
            for hdu in hdul:
                if hdu.data is not None and hasattr(hdu.data, 'shape'):
                    arr = np.array(hdu.data, dtype=np.float32)
                    if arr.ndim == 3:
                        arr = arr[0]
                    if arr.ndim == 2:
                        data = arr
                        break
        if data is None:
            st.error("No 2D image data found in this file.")
            st.stop()
        data = data[:512, :512] if data.shape[0] > 512 else data
        data -= np.median(data)
        img = np.clip(data, 0, None)
    else:
        st.info("👈 Upload a FITS file to get started.")
        st.stop()

elif source == "Use sample images":
    sample = st.sidebar.selectbox("Choose a sample", [
        "Synthetic Star Field",
        "Synthetic Galaxy + Nebula",
        "Synthetic Dense Cluster"
    ])

    @st.cache_data
    def make_star_field(seed, num_stars, nebula=False, dense=False):
        np.random.seed(seed)
        size = 512
        image = np.zeros((size, size), dtype=np.float32)

        # Stars
        for _ in range(num_stars):
            x, y = np.random.randint(10, size-10, 2)
            brightness = np.random.exponential(500 if not dense else 200)
            for dx in range(-8, 9):
                for dy in range(-8, 9):
                    if 0 <= x+dx < size and 0 <= y+dy < size:
                        image[x+dx, y+dy] += brightness * np.exp(-(dx**2+dy**2)/3.0)

        # Optional nebula
        if nebula:
            cx, cy = size//2, size//2
            for i in range(size):
                for j in range(size):
                    dist = np.sqrt((i-cx)**2 + (j-cy)**2)
                    image[i,j] += 80 * np.exp(-dist**2 / (2*60**2))

        # Noise
        image = np.random.poisson(np.clip(image, 0, None)).astype(np.float32)
        image += np.random.normal(0, 15, image.shape).astype(np.float32)

        # Cosmic rays
        for _ in range(15):
            x, y = np.random.randint(0, size, 2)
            image[x, y] += np.random.uniform(2000, 8000)

        return np.clip(image, 0, None)

    if sample == "Synthetic Star Field":
        img = make_star_field(seed=42, num_stars=200)
    elif sample == "Synthetic Galaxy + Nebula":
        img = make_star_field(seed=7, num_stars=150, nebula=True)
    elif sample == "Synthetic Dense Cluster":
        img = make_star_field(seed=99, num_stars=500, dense=True)

def norm(img):
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(img)
    return np.clip((img - vmin) / (vmax - vmin), 0, 1)

def make_psf(size=15, sigma=2.0):
    x = np.arange(size) - size // 2
    xx, yy = np.meshgrid(x, x)
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return psf / psf.sum()

def snr(img):
    noise = estimate_sigma(img)
    signal = np.mean(img[img > np.median(img)])
    return signal / noise if noise > 0 else 0

# --- Sidebar ---
st.sidebar.header("⚙️ Controls")
method = st.sidebar.selectbox("Denoising Method", [
    "Gaussian Blur",
    "Median Filter",
    "Richardson-Lucy Deconvolution"
])

if method == "Gaussian Blur":
    sigma = st.sidebar.slider("Sigma (blur strength)", 0.5, 5.0, 1.5)
    result = gaussian(img, sigma=sigma)

elif method == "Median Filter":
    radius = st.sidebar.slider("Kernel radius (px)", 1, 5, 2)
    result = median(img, disk(radius))

elif method == "Richardson-Lucy Deconvolution":
    psf_sigma = st.sidebar.slider("PSF sigma (seeing)", 0.5, 4.0, 2.0)
    iterations = st.sidebar.slider("Iterations", 5, 50, 20)
    psf = make_psf(sigma=psf_sigma)
    with st.spinner("Running deconvolution..."):
        result = richardson_lucy(img, psf, num_iter=iterations)

# --- Display ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Original")
    st.image(norm(img), clamp=True, use_container_width=True)
    st.metric("SNR", f"{snr(img):.2f}")

with col2:
    st.subheader(f"Denoised ({method})")
    st.image(norm(result), clamp=True, use_container_width=True)
    st.metric("SNR", f"{snr(result):.2f}",
              delta=f"{((snr(result)-snr(img))/snr(img)*100):.1f}%")

# --- Histogram ---
st.subheader("Pixel Intensity Distribution")
fig, ax = plt.subplots(figsize=(10, 3))
ax.hist(img.flatten(), bins=150, alpha=0.5, label="Original", color="steelblue")
ax.hist(result.flatten(), bins=150, alpha=0.5, label="Denoised", color="coral")
ax.set_xlabel("Pixel intensity")
ax.set_ylabel("Count")
ax.legend()
st.pyplot(fig)

# --- Physics explainer ---
with st.expander("📖 How does this work?"):
    st.markdown("""
    **Noise sources in astronomical images:**
    - **Photon shot noise** — light is quantised; faint sources are noisy by nature (Poisson statistics)
    - **Read noise** — electronic noise from the sensor readout
    - **Cosmic rays** — high-energy particles leave sharp spikes on the detector
    - **Atmospheric seeing** — turbulence blurs stars into a spread called the PSF
    
    **Richardson-Lucy deconvolution** mathematically inverts the blurring by modelling 
    the Point Spread Function and iteratively recovering the true image.
    """)