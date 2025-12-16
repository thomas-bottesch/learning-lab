/*
Minimal C program that loads an ONNX model with ONNX Runtime C API
and runs inference over the MNIST test set stored in IDX format.
*/ 

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <onnxruntime/core/session/onnxruntime_c_api.h>

static const OrtApi* g_ort = NULL;

static int32_t read_int32_be(FILE* f) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) return -1;
    return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
}

static unsigned char* load_mnist_images(const char* path, int* out_count, 
                                        int* out_rows, int* out_cols) {
    FILE* f = NULL;
    unsigned char* data = NULL;
    int magic = 0;
    int count = 0;
    int rows = 0;
    int cols = 0;
    size_t total = 0;

    f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open images file: %s\n", path);
        return NULL;
    }
    magic = read_int32_be(f);
    count = read_int32_be(f);
    rows = read_int32_be(f);
    cols = read_int32_be(f);
    if (magic != 2051) {
        fprintf(stderr, "Invalid magic number in images file: %d\n", magic);
        fclose(f);
        return NULL;
    }
    total = (size_t)count * rows * cols;
    data = (unsigned char*)malloc(total);
    if (!data) {
        fclose(f);
        return NULL;
    }
    if (fread(data, 1, total, f) != total) {
        fprintf(stderr, "Failed to read images\n");
        free(data);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *out_count = count;
    *out_rows = rows;
    *out_cols = cols;
    return data;
}

static unsigned char* load_mnist_labels(const char* path, int* out_count) {
    FILE* f = NULL;
    unsigned char* data = NULL;
    int magic = 0;
    int count = 0;

    f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open labels file: %s\n", path);
        return NULL;
    }
    magic = read_int32_be(f);
    count = read_int32_be(f);
    if (magic != 2049) {
        fprintf(stderr, "Invalid magic number in labels file: %d\n", magic);
        fclose(f);
        return NULL;
    }
    data = (unsigned char*)malloc(count);
    if (!data) {
        fclose(f);
        return NULL;
    }
    if (fread(data, 1, count, f) != (size_t)count) {
        fprintf(stderr, "Failed to read labels\n");
        free(data);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *out_count = count;
    return data;
}

int main(int argc, char** argv) {
    /* Pointers */
    const char *model_path  = "custom_onnx_runtime/model.ort";
    const char *images_path = "data/MNIST/raw/t10k-images-idx3-ubyte";
    const char *labels_path = "data/MNIST/raw/t10k-labels-idx1-ubyte";

    char* input_name = NULL;
    char* output_name = NULL;
    unsigned char* images = NULL;
    unsigned char* labels = NULL;
    const char* input_names[1];
    const char* output_names[1];
    const OrtValue* input_values[1];
    float* outarr = NULL;
    FILE* csv_file = NULL;

    /* ONNX Runtime objects */
    OrtEnv* env = NULL;
    OrtSessionOptions* session_options = NULL;
    OrtSession* session = NULL;
    OrtAllocator* allocator = NULL;
    OrtMemoryInfo* mem_info = NULL;
    OrtStatus* status = NULL;
    OrtValue* input_tensor = NULL;
    OrtValue* output_tensor = NULL;

    /* Scalars */
    int img_count = 0;
    int rows = 0;
    int cols = 0;
    int lbl_count = 0;
    int i = 0;
    int k = 0;
    int best = 0;
    int true_label = 0;

    /* Arrays */
    int64_t input_shape[4];

    /* Sizes */
    size_t input_size = 0;
    size_t correct = 0;
    size_t mismatches_reported = 0;

    /* Floats */
    float bestv = 0.0f;

    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "mnist_infer", &env);
    if (status != NULL) {
        fprintf(stderr, "CreateEnv failed: %s\n", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        return 1;
    }

    status = g_ort->CreateSessionOptions(&session_options);
    if (status != NULL) {
        fprintf(stderr, "CreateSessionOptions failed: %s\n", 
                g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseEnv(env);
        return 1;
    }
    /* optional: set intra op threads, optimization level, etc. */

    status = g_ort->CreateSession(env, model_path, session_options, &session);
    if (status != NULL) {
        fprintf(stderr, "CreateSession failed: %s\n", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSessionOptions(session_options);
        g_ort->ReleaseEnv(env);
        return 1;
    }

    status = g_ort->GetAllocatorWithDefaultOptions(&allocator);
    if (status != NULL) {
        fprintf(stderr, "GetAllocatorWithDefaultOptions failed: %s\n", 
                g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSession(session);
        g_ort->ReleaseSessionOptions(session_options);
        g_ort->ReleaseEnv(env);
        return 1;
    }
    printf("GetAllocatorWithDefaultOptions -> allocator=%p\n", (void*)allocator);

    status = g_ort->SessionGetInputName(session, 0, allocator, &input_name);
    if (status != NULL) {
        fprintf(stderr, "SessionGetInputName failed: %s\n", 
                g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseAllocator(allocator);
        g_ort->ReleaseSession(session);
        g_ort->ReleaseSessionOptions(session_options);
        g_ort->ReleaseEnv(env);
        return 1;
    }
    status = g_ort->SessionGetOutputName(session, 0, allocator, &output_name);
    if (status != NULL) {
        fprintf(stderr, "SessionGetOutputName failed: %s\n", 
                g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        status = g_ort->AllocatorFree(allocator, input_name); /* best-effort free */
        if (status != NULL) {
            fprintf(stderr, "AllocatorFree(input_name) failed: %s\n", 
                    g_ort->GetErrorMessage(status));
            g_ort->ReleaseStatus(status);
        }
        g_ort->ReleaseAllocator(allocator);
        g_ort->ReleaseSession(session);
        g_ort->ReleaseSessionOptions(session_options);
        g_ort->ReleaseEnv(env);
        return 1;
    }

    images = load_mnist_images(images_path, &img_count, &rows, &cols);
    if (!images) return 1;
    labels = load_mnist_labels(labels_path, &lbl_count);
    if (!labels) {
        free(images);
        return 1;
    }
    if (img_count != lbl_count) {
        fprintf(stderr, "Image/label count mismatch\n");
        free(images);
        free(labels);
        return 1;
    }

    status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, 
                                        &mem_info);
    if (status != NULL) {
        fprintf(stderr, "CreateCpuMemoryInfo failed: %s\n", 
                g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        status = g_ort->AllocatorFree(allocator, input_name);
        if (status != NULL) {
            fprintf(stderr, "AllocatorFree(input_name) failed: %s\n", 
                    g_ort->GetErrorMessage(status));
            g_ort->ReleaseStatus(status);
        }
        status = g_ort->AllocatorFree(allocator, output_name);
        if (status != NULL) {
            fprintf(stderr, "AllocatorFree(output_name) failed: %s\n", 
                    g_ort->GetErrorMessage(status));
            g_ort->ReleaseStatus(status);
        }
        g_ort->ReleaseSession(session);
        g_ort->ReleaseSessionOptions(session_options);
        g_ort->ReleaseEnv(env);
        free(images);
        free(labels);
        return 1;
    }

    input_shape[0] = 1;
    input_shape[1] = 1;
    input_shape[2] = 28;
    input_shape[3] = 28;
    input_size = (size_t)1 * 1 * 28 * 28;

    correct = 0;
    mismatches_reported = 0;

    /* Open CSV file for writing predictions */
    csv_file = fopen("c_predictions.csv", "w");
    if (!csv_file) {
        fprintf(stderr, "Failed to open CSV file for writing\n");
        free(images);
        free(labels);
        g_ort->ReleaseMemoryInfo(mem_info);
        status = g_ort->AllocatorFree(allocator, input_name);
        if (status != NULL) g_ort->ReleaseStatus(status);
        status = g_ort->AllocatorFree(allocator, output_name);
        if (status != NULL) g_ort->ReleaseStatus(status);
        g_ort->ReleaseSession(session);
        g_ort->ReleaseSessionOptions(session_options);
        g_ort->ReleaseEnv(env);
        return 1;
    }
    fprintf(csv_file, "index,prediction,label,top_logit\n");

    for (i = 0; i < img_count; ++i) {
        unsigned char* img_ptr = images + (size_t)i * input_size;

        input_tensor = NULL;
        /* Create tensor that uses the image data buffer directly */
        status = g_ort->CreateTensorWithDataAsOrtValue(mem_info, (void*)img_ptr, 
                                                       input_size, input_shape, 
                                                       4, 
                                                       ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, 
                                                       &input_tensor);
        if (status != NULL) {
            fprintf(stderr, "CreateTensorWithDataAsOrtValue failed at index %d: %s\n", 
                    i, g_ort->GetErrorMessage(status));
            g_ort->ReleaseStatus(status);
            break;
        }

        input_names[0] = input_name;
        input_values[0] = input_tensor;
        output_names[0] = output_name;
        output_tensor = NULL;

        status = g_ort->Run(session, NULL, input_names, input_values, 1, 
                            output_names, 1, &output_tensor);
        if (status != NULL) {
            fprintf(stderr, "Run() failed at index %d: %s\n", 
                    i, g_ort->GetErrorMessage(status));
            g_ort->ReleaseStatus(status);
            g_ort->ReleaseValue(input_tensor);
            break;
        }

        outarr = NULL;
        status = g_ort->GetTensorMutableData(output_tensor, (void**)&outarr);
        if (status != NULL) {
            fprintf(stderr, "GetTensorMutableData failed at index %d: %s\n", 
                    i, g_ort->GetErrorMessage(status));
            g_ort->ReleaseStatus(status);
            g_ort->ReleaseValue(output_tensor);
            g_ort->ReleaseValue(input_tensor);
            break;
        }
        best = 0;
        bestv = outarr[0];
        for (k = 1; k < 10; ++k) {
            if (outarr[k] > bestv) {
                bestv = outarr[k];
                best = k;
            }
        }
        true_label = labels[i];
        
        /* Write to CSV */
        fprintf(csv_file, "%d,%d,%d,%.6f\n", i, best, true_label, bestv);
        
        if (best == true_label) ++correct;
        else {
            if (mismatches_reported < 10) {
                printf("Mismatch idx %d: pred=%d label=%d (logit=%.6f)\n", i, best, true_label, bestv);
                mismatches_reported++;
            }
        }

        g_ort->ReleaseValue(output_tensor);
        g_ort->ReleaseValue(input_tensor);
    }

    fclose(csv_file);
    printf("Predictions written to c_predictions.csv\n");

    printf("Processed %d images. Accuracy: %.2f%% (%lu/%d)\n", img_count, 
           (double)correct * 100.0 / (double)img_count, 
           (unsigned long)correct, img_count);
    printf("GetAllocatorWithDefaultOptions -> allocator=%p\n", (void*)allocator);

    /* cleanup */
    free(images);
    free(labels);
    if (mem_info) {
        g_ort->ReleaseMemoryInfo(mem_info);
        mem_info = NULL;
    }
    if (input_name != NULL && allocator != NULL) {
        status = g_ort->AllocatorFree(allocator, input_name);
        if (status != NULL) {
            fprintf(stderr, "AllocatorFree(input_name) failed: %s\n",
                    g_ort->GetErrorMessage(status));
            g_ort->ReleaseStatus(status);
            status = NULL;
        }
        input_name = NULL;
    }
    status = g_ort->AllocatorFree(allocator, output_name);
    if (status != NULL) {
        fprintf(stderr, "AllocatorFree(output_name) failed: %s\n", 
                g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
    }

    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);
    return 0;
}