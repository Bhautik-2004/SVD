import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import time

# Page configuration
st.set_page_config(
    page_title="SVD Image Compression Tool",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("Image Compression with SVD")

# Function to check image size before processing
def check_image_size(image):
    width, height = image.size
    num_pixels = width * height

    if num_pixels > 4000000:  # 4 megapixels
        return False, f"Image size ({width}×{height}, {num_pixels:,} pixels) is too large. Please use an image smaller than 2000×2000 pixels."
    return True, ""


# SVD compression function with better error handling
def svd_compress(image, k, status_container):
    try:
        # Convert image to NumPy array
        img_array = np.array(image)

        # Store original size for comparison
        original_bytes = img_array.nbytes

        # Handle different image formats
        if len(img_array.shape) == 2:
            # Grayscale image
            U, S, Vt = np.linalg.svd(img_array, full_matrices=False)

            # Keep only k singular values
            compressed = np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))
            compressed = np.clip(compressed, 0, 255).astype(np.uint8)

        elif len(img_array.shape) == 3:
            if img_array.shape[2] == 4:
                rgb = img_array[:, :, :3]
                alpha = img_array[:, :, 3]

                channels = []
                for i in range(3):  # Process RGB channels
                    channel = rgb[:, :, i]
                    U, S, Vt = np.linalg.svd(channel, full_matrices=False)
                    compressed_channel = np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))
                    channels.append(compressed_channel)

                # Combine RGB channels and add back alpha
                compressed_rgb = np.stack(channels, axis=2)
                compressed_rgb = np.clip(compressed_rgb, 0, 255).astype(np.uint8)
                alpha = alpha.reshape(alpha.shape[0], alpha.shape[1], 1)
                compressed = np.concatenate((compressed_rgb, alpha), axis=2)

            else:
                # Standard RGB image
                channels = []
                for i in range(img_array.shape[2]):
                    channel = img_array[:, :, i]
                    U, S, Vt = np.linalg.svd(channel, full_matrices=False)
                    compressed_channel = np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))
                    channels.append(compressed_channel)

                compressed = np.stack(channels, axis=2)
                compressed = np.clip(compressed, 0, 255).astype(np.uint8)

        # Calculate compression metrics
        compressed_bytes = k * (img_array.shape[0] + img_array.shape[1])
        if len(img_array.shape) == 3:
            compressed_bytes *= img_array.shape[2]

        compression_ratio = original_bytes / compressed_bytes

        return compressed, compression_ratio, None

    except Exception as e:
        return None, 0, f"Error during compression: {str(e)}"


# Function to plot singular values
def plot_singular_values(image):
    img_array = np.array(image)

    fig, ax = plt.subplots(figsize=(8, 4))

    if len(img_array.shape) == 2:
        # Grayscale
        _, S, _ = np.linalg.svd(img_array, full_matrices=False)
        ax.plot(S[:100], color='black', label='Grayscale')
    else:
        # RGB
        colors = ['red', 'green', 'blue']
        for i in range(min(3, img_array.shape[2])):
            _, S, _ = np.linalg.svd(img_array[:, :, i], full_matrices=False)
            ax.plot(S[:100], color=colors[i], label=colors[i].capitalize())

    ax.set_title('Singular Value Distribution')
    ax.set_xlabel('Index')
    ax.set_ylabel('Magnitude')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig


# Create sidebar for controls
st.sidebar.header("Upload & Settings")

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload an image (JPG, PNG)",
    type=["jpg", "jpeg", "png"],
    help="Maximum recommended size: 2000×2000 pixels"
)

# Main content area
if uploaded_file is not None:
    # Status container for messages
    status_container = st.empty()

    try:
        # Load image
        status_container.info("Loading image...")
        image = Image.open(uploaded_file)

        # Check image size
        size_ok, size_message = check_image_size(image)
        if not size_ok:
            status_container.warning(size_message)
            st.warning("⚠️ Processing may be slow. Consider using a smaller image for better performance.")

        # Display original image stats
        file_size = uploaded_file.size / 1024  # KB
        width, height = image.size
        st.sidebar.success(f"✅ Image loaded successfully")
        st.sidebar.info(f"📏 Dimensions: {width} × {height} pixels")
        st.sidebar.info(f"📦 Original file size: {file_size:.1f} KB")

        # Create column layout
        col1, col2 = st.columns(2)

        # Show original image
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        # Calculate maximum k value based on image dimensions
        max_singular = min(width, height)
        default_k = min(int(max_singular * 0.1), 50)  # Default to 10% or 50, whichever is smaller

        # K slider
        k = st.sidebar.slider(
            "Compression level (k)",
            min_value=1,
            max_value=min(max_singular, 100),
            value=default_k,
            help="Higher values = better quality but less compression"
        )

        # Add details expander in sidebar
        with st.sidebar.expander("What is k?"):
            st.markdown("""
                **k** represents the number of singular values used in the compression.

                - **Lower k**: Higher compression, lower quality
                - **Higher k**: Lower compression, higher quality

                Recommended starting point: 10-20% of the smallest image dimension.
            """)

        # Process button
        if st.sidebar.button("Compress Image", type="primary"):
            # Show status
            status_container.info("Compressing image...")
            progress_bar = st.sidebar.progress(0)

            # Start compression with timing
            start_time = time.time()

            # Process image
            compressed_img, compression_ratio, error = svd_compress(image, k, status_container)

            # Update progress
            for i in range(10):
                progress_bar.progress((i + 1) * 10)
                time.sleep(0.05)

            processing_time = time.time() - start_time

            # Check for errors
            if error:
                status_container.error(error)
            else:
                # Clear status
                status_container.empty()

                # Display compressed image
                with col2:
                    st.subheader(f"Compressed Image (k={k})")
                    st.image(compressed_img, use_container_width=True)

                    # Display compression metrics
                    metrics_col1, metrics_col2 = st.columns(2)
                    with metrics_col1:
                        st.metric("Compression Ratio", f"{compression_ratio:.1f}x")
                    with metrics_col2:
                        st.metric("Processing Time", f"{processing_time:.2f}s")

                    # Memory consumption comparison
                    original_mem = np.prod(np.array(image).shape)
                    compressed_mem = k * (width + height)
                    if len(np.array(image).shape) == 3:
                        compressed_mem *= np.array(image).shape[2]

                    memory_savings = (1 - compressed_mem / original_mem) * 100
                    st.success(f"💾 Memory savings: {memory_savings:.1f}%")

                    # Convert to PIL Image for downloading
                    pil_compressed = Image.fromarray(compressed_img)
                    buf = io.BytesIO()
                    pil_compressed.save(buf, format="PNG")

                    # Download button
                    st.download_button(
                        label="Download Compressed Image",
                        data=buf.getvalue(),
                        file_name=f"compressed_k{k}.png",
                        mime="image/png"
                    )

                # Show singular value plot
                st.subheader("Singular Value Analysis")
                fig = plot_singular_values(image)
                st.pyplot(fig)

                # Add explanation
                with st.expander("Understanding this plot"):
                    st.markdown("""
                        The singular values plot shows the importance of each component in the image.

                        - The x-axis represents the index of singular values
                        - The y-axis (logarithmic scale) shows their magnitude
                        - The rapid drop indicates that most information is contained in the first few values
                        - Your compression level (k) determines how many values are kept
                    """)

                # Visual comparison of different k values
                st.subheader("Visual Compression Spectrum")
                st.markdown("Compare different compression levels")

                # Set 5 different k values
                test_k_values = [1, 5, 10, 25, 50]
                test_k_values = [k for k in test_k_values if k <= min(width, height)]

                # Create columns for different k values
                k_cols = st.columns(len(test_k_values))

                # Process and display each k value
                for i, test_k in enumerate(test_k_values):
                    test_img, test_ratio, _ = svd_compress(image, test_k, st.empty())
                    with k_cols[i]:
                        st.image(test_img, caption=f"k={test_k}", use_container_width=True)
                        st.text(f"Ratio: {test_ratio:.1f}x")

    except Exception as e:
        status_container.error(f"Error processing image: {str(e)}")
        st.error("Please try with a different image.")

else:
    # Display instructions if no image is uploaded
    st.info("👈 Please upload an image from the sidebar to begin")

    # Show example visuals and explanation
    st.subheader("How SVD Compression Works")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        Singular Value Decomposition (SVD) is a powerful technique for image compression:

        1. **Breaking down the image:** SVD decomposes each color channel into three matrices (U, Σ, V^T)
        2. **Keeping what matters:** By retaining only the most significant singular values (k), we can approximate the original image
        3. **Balancing quality and size:** The 'k' parameter lets you control the trade-off between image quality and file size

        SVD works especially well for:
        - Images with smooth gradients
        - Photos with clear backgrounds
        - Images where some quality loss is acceptable
        """)

    with col2:
        # Display example image of SVD decomposition
        st.image("https://miro.medium.com/max/640/1*6dOWJYMOKnL3LIZ0iYigZA.png",
                 caption="SVD decomposition illustration",
                 use_container_width=True)

    # Show warning about limitations
    st.warning("""
    ⚠️ **Limitations:**
    - Not suitable for text-heavy images where detail is critical
    - Not recommended for images that require perfect reproduction
    - Large images (>2000×2000 pixels) may take significant processing time
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("SVD Image Compression Tool | VB")
