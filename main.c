#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Hiperparametreler
#define INPUT_SIZE 784       // mnist veri seti için 28x28 piksel
#define MAX_EPOCHS 100       // Adam ve SGD için yakınsar ancak GD için daha yüksek epoch gerekebilir
#define LEARNING_RATE 0.01
#define BETA1 0.9
#define BETA2 0.999
#define EPSILON 1e-8
#define MAX_LINE_LENGTH 10000
#define MAX_SAMPLES 10000 


// Adam Optimization Structure
typedef struct {
    double *weights;       
    double *m;             // İlk moment vektörü
    double *v;             // İkinci moment vektörü
    int input_count;       
} AdamOptimizer;

// Data Structure
typedef struct {
    double **data;         
    double *labels;        
    int num_samples;       
} Dataset;

// SGD Optimization Structure
typedef struct {
    double *weights;       
    double learning_rate;  
    int input_count;       
} SGDOptimizer;

// Gradient Descent Optimization Structure
typedef struct {
    double *weights;       
    double learning_rate;  
    int input_count;       
} GDOptimizer;

// Her epoch için ağırlıkları csv dosyasına kaydeder
void log_weights_to_csv(FILE* file, int epoch, double* weights, int size) {
    fprintf(file, "%d", epoch);  
    for (int i = 0; i < size; i++) {
        fprintf(file, ",%f", weights[i]);  
    }
    fprintf(file, "\n");  
}

// CSV dosyasını (mnist) okuyun ve eğitim ve test veri setlerine ayırın. Eğitim için yüzde 80.
// CSV'nin ilk sütunu etikettir ve ayrılmıştır, ayrıca gri tonlamalıdır ve [0,1] aralığına normalize edilmiştir
void read_and_split_csv(const char* filename, Dataset** train_dataset, Dataset** test_dataset) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file %s\n", filename);
        return;
    }

    double** all_data = malloc(MAX_SAMPLES * sizeof(double*));
    double* all_labels = malloc(MAX_SAMPLES * sizeof(double));
    int num_samples = 0;

    char line[MAX_LINE_LENGTH];
    fgets(line, sizeof(line), file);  

    while (fgets(line, sizeof(line), file) && num_samples < MAX_SAMPLES) {
        all_data[num_samples] = malloc(INPUT_SIZE * sizeof(double));
        
        char* token = strtok(line, ",");
        all_labels[num_samples] = atof(token);  
        
        for (int i = 0; i < INPUT_SIZE; i++) {
            token = strtok(NULL, ",");
            all_data[num_samples][i] = atof(token) / 255.0;  
        }

        num_samples++;
    }
    fclose(file);

    // Verileri eğitim ve test setlerine ayırır
    int train_size = (int)(num_samples * 0.8);
    int test_size = num_samples - train_size;

    *train_dataset = malloc(sizeof(Dataset));
    *test_dataset = malloc(sizeof(Dataset));

    (*train_dataset)->data = malloc(train_size * sizeof(double*));
    (*train_dataset)->labels = malloc(train_size * sizeof(double));
    (*train_dataset)->num_samples = train_size;

    (*test_dataset)->data = malloc(test_size * sizeof(double*));
    (*test_dataset)->labels = malloc(test_size * sizeof(double));
    (*test_dataset)->num_samples = test_size;

    for (int i = 0; i < train_size; i++) {
        (*train_dataset)->data[i] = all_data[i];
        (*train_dataset)->labels[i] = all_labels[i];
    }

    for (int i = 0; i < test_size; i++) {
        (*test_dataset)->data[i] = all_data[train_size + i];
        (*test_dataset)->labels[i] = all_labels[train_size + i];
    }

    free(all_data);
    free(all_labels);
}


void free_dataset(Dataset* dataset) {
    if (!dataset) return;
    for (int i = 0; i < dataset->num_samples; i++) {
        free(dataset->data[i]);
    }
    free(dataset->data);
    free(dataset->labels);
    free(dataset);
}

// Normal dağılım ile Gradient Descent Optimizer'ı başlatır
GDOptimizer* create_gd_optimizer(int input_count, double learning_rate) {
    GDOptimizer* optimizer = malloc(sizeof(GDOptimizer));
    optimizer->input_count = input_count;
    optimizer->learning_rate = learning_rate;
    optimizer->weights = calloc(input_count + 1, sizeof(double));
    
    // rand kütüphanesini zaman seed'le kullanır
    srand(time(NULL));
    for (int i = 0; i < input_count + 1; i++) {
        optimizer->weights[i] = ((double)rand() / RAND_MAX) * 0.1 - 0.05;
    }
    
    return optimizer;
}

// Ağırlıkların girişlerle (veri seti) çarpımının tanh'ını döndürerek çıktıyı tahmin eder
double predict_gd(GDOptimizer* optimizer, double* input) {
    double weighted_sum = optimizer->weights[0];  // Bias term
    
    for (int i = 0; i < optimizer->input_count; i++) {
        weighted_sum += optimizer->weights[i + 1] * input[i];
    }
    
    return tanh(weighted_sum);
}

// Gradient Descent kullanarak eğitir: gradient descent formülü kullanılarak optimize edilmiştir
void train_gd(GDOptimizer* optimizer, Dataset* dataset, const char* csv_filename, const char* csv_filename2, int iterations) {
    FILE* file = fopen(csv_filename, "w");
    if (!file) {
        printf("Error\n", csv_filename);
        return;
    }

    fprintf(file, "Epoch");
    for (int i = 0; i <= optimizer->input_count; i++) {
        fprintf(file, ",w%d", i);
    }
    fprintf(file, "\n");

    FILE* file2 = fopen(csv_filename2, "w");
    if (!file2) {
        printf("Error\n", csv_filename2);
        return;
    }

    fprintf(file2, "Epoch,Time,Loss\n");
    double cumulative_time = 0.0;

    // epoch 0 için başlangıç ağırlıklarını ve metrikleri yazın
    double initial_loss = 0.0;
    for (int i = 0; i < dataset->num_samples; i++) {
        double prediction = predict_gd(optimizer, dataset->data[i]);
        double true_label = dataset->labels[i];
        double error = true_label - prediction;
        initial_loss += error * error;
    }
    initial_loss /= dataset->num_samples;
    fprintf(file2, "0,0.000000,%f\n", initial_loss);
    log_weights_to_csv(file, 0, optimizer->weights, optimizer->input_count + 1);

    printf("Epoch 0: Loss = %f, Cumulative Time = 0.000000 seconds\n", initial_loss);

    for (int epoch = 1; epoch <= iterations; epoch++) {
        clock_t start_time = clock();

        double total_error = 0.0;
        double* gradients = calloc(optimizer->input_count + 1, sizeof(double));

        for (int i = 0; i < dataset->num_samples; i++) {
            double prediction = predict_gd(optimizer, dataset->data[i]);
            double true_label = dataset->labels[i];
            double error = true_label - prediction;
            total_error += error * error;
            // tanh'ın türevi olan 1 - tanh^2
            double gradient = error * (1 - prediction * prediction);  
            gradients[0] += gradient;  
            for (int j = 0; j < optimizer->input_count; j++) {
                gradients[j + 1] += gradient * dataset->data[i][j];
            }
        }

        for (int j = 0; j <= optimizer->input_count; j++) {
            // gradient descent formülü
            optimizer->weights[j] += optimizer->learning_rate * gradients[j] / dataset->num_samples;
        }

        free(gradients);

        clock_t end_time = clock();
        double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        cumulative_time += elapsed_time;

        double avg_loss = total_error / dataset->num_samples;

        fprintf(file2, "%d,%f,%f\n", epoch, cumulative_time, avg_loss);
        log_weights_to_csv(file, epoch, optimizer->weights, optimizer->input_count + 1);

        if (epoch % 10 == 0 || epoch == iterations) {
            printf("Epoch %d: Loss = %f, Cumulative Time = %f seconds\n", epoch, avg_loss, cumulative_time);
        }
    }

    fclose(file);
    fclose(file2);
}


void free_gd_optimizer(GDOptimizer* optimizer) {
    if (!optimizer) return;
    free(optimizer->weights);
    free(optimizer);
}

// GD ağırlıklarını kullanarak modeli test eder
void test_model_gd(GDOptimizer* optimizer, Dataset* test_dataset) {
    double total_accuracy = 0.0;
    double total_error = 0.0;

    for (int i = 0; i < test_dataset->num_samples; i++) {
        double prediction = predict_gd(optimizer, test_dataset->data[i]);
        double true_label = test_dataset->labels[i];

        double error = true_label - prediction;
        total_error += error * error;

        total_accuracy += (fabs(round(prediction) - true_label) < 0.5) ? 1.0 : 0.0;
    }

    printf("\nTest Results (GD):\n");
    printf("Test Error: %f\n", total_error / test_dataset->num_samples);
    printf("Test Accuracy: %f%%\n", (total_accuracy / test_dataset->num_samples) * 100);
}

// Normal dağılım ile Adam Optimizer'ı başlatır
AdamOptimizer* create_adam_optimizer(int input_count) {
    AdamOptimizer* optimizer = malloc(sizeof(AdamOptimizer));
    optimizer->input_count = input_count;
    
    optimizer->weights = calloc(input_count + 1, sizeof(double));
    optimizer->m = calloc(input_count + 1, sizeof(double));
    optimizer->v = calloc(input_count + 1, sizeof(double));
    
    // Rastgele ağırlık başlatma
    srand(time(NULL));
    for (int i = 0; i < input_count + 1; i++) {
        optimizer->weights[i] = ((double)rand() / RAND_MAX) * 0.1 - 0.05;
    }
    
    return optimizer;
}

void free_optimizer(AdamOptimizer* optimizer) {
    if (!optimizer) return;
    free(optimizer->weights);
    free(optimizer->m);
    free(optimizer->v);
    free(optimizer);
}

// Normal dağılım ile SGD Optimizer'ı başlatır
SGDOptimizer* create_sgd_optimizer(int input_count, double learning_rate) {
    SGDOptimizer* optimizer = malloc(sizeof(SGDOptimizer));
    optimizer->input_count = input_count;
    optimizer->learning_rate = learning_rate;
    optimizer->weights = calloc(input_count + 1, sizeof(double));
    
    // Rastgele ağırlık başlatma
    srand(time(NULL));
    for (int i = 0; i < input_count + 1; i++) {
        optimizer->weights[i] = ((double)rand() / RAND_MAX) * 0.1 - 0.05;
    }
    
    return optimizer;
}


double predict_sgd(SGDOptimizer* optimizer, double* input) {
    double weighted_sum = optimizer->weights[0];  
    
    for (int i = 0; i < optimizer->input_count; i++) {
        weighted_sum += optimizer->weights[i + 1] * input[i];
    }
    
    return tanh(weighted_sum);
}
 
void train_sgd(SGDOptimizer* optimizer, Dataset* dataset, const char* csv_filename, const char* csv_filename2, int iterations) {
    
    FILE* file = fopen(csv_filename, "w");
    if (!file) {
        printf("Error\n", csv_filename);
        return;
    }

    
    fprintf(file, "Epoch");
    for (int i = 0; i <= optimizer->input_count; i++) {
        fprintf(file, ",w%d", i);
    }
    fprintf(file, "\n");

    FILE* file2 = fopen(csv_filename2, "w");
    if (!file) {
        printf("Error\n", csv_filename2);
        return;
    }

    fprintf(file2, "Epoch,Time,Loss\n");
    double cumulative_time = 0.0;  

    
    double initial_loss = 0.0;
    for (int i = 0; i < dataset->num_samples; i++) {
        double prediction = predict_sgd(optimizer, dataset->data[i]);
        double true_label = dataset->labels[i];
        double error = true_label - prediction;
        initial_loss += error * error;
    }
    initial_loss /= dataset->num_samples;
    fprintf(file2, "0,0.000000,%f\n", initial_loss);
    log_weights_to_csv(file, 0, optimizer->weights, optimizer->input_count + 1);

    printf("Epoch 0: Loss = %f, Cumulative Time = 0.000000 seconds\n", initial_loss);

    for (int epoch = 1; epoch <= iterations; epoch++) {
        clock_t start_time = clock();  

        double total_error = 0.0;

        // Verileri karıştırir
        for (int i = 0; i < dataset->num_samples; i++) {
            int j = rand() % dataset->num_samples;

            
            double* temp_data = dataset->data[i];
            dataset->data[i] = dataset->data[j];
            dataset->data[j] = temp_data;

            double temp_label = dataset->labels[i];
            dataset->labels[i] = dataset->labels[j];
            dataset->labels[j] = temp_label;
        }

        for (int i = 0; i < dataset->num_samples; i++) {
            double prediction = predict_sgd(optimizer, dataset->data[i]);
            double true_label = dataset->labels[i];

            double error = true_label - prediction;
            total_error += error * error;

            // tanh'ın türevi olan 1 - tanh^2
            double gradient = error * (1 - prediction * prediction);  

            for (int j = 0; j < optimizer->input_count + 1; j++) {
                double grad = (j == 0) ? gradient : gradient * dataset->data[i][j - 1];
                optimizer->weights[j] += optimizer->learning_rate * grad;
            }
        }

        clock_t end_time = clock();  
        double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;  
        cumulative_time += elapsed_time;

        double avg_loss = total_error / dataset->num_samples;  

        
        fprintf(file2, "%d,%f,%f\n", epoch, cumulative_time, avg_loss);

        log_weights_to_csv(file, epoch, optimizer->weights, optimizer->input_count + 1);

        
        if (epoch % 10 == 0 || epoch == iterations - 1) {
            printf("Epoch %d: Loss = %f, Cumulative Time = %f seconds\n",
                   epoch, avg_loss, cumulative_time);
        }
    }

    fclose(file);
    fclose(file2);
}


void free_sgd_optimizer(SGDOptimizer* optimizer) {
    if (!optimizer) return;
    free(optimizer->weights);
    free(optimizer);
}

// Tahmin
double predict(AdamOptimizer* optimizer, double* input) {
    double weighted_sum = optimizer->weights[0];  // Bias term
    
    for (int i = 0; i < optimizer->input_count; i++) {
        weighted_sum += optimizer->weights[i + 1] * input[i];
    }
    
    return tanh(weighted_sum);
}

// Adam kullanarak eğitir: Adam formülü kullanılarak optimize edilmiştir
void train_adam(AdamOptimizer* optimizer, Dataset* dataset, const char* csv_filename, const char* csv_filename2, int iterations) {
    int t = 0;

    FILE* file = fopen(csv_filename, "w");
    if (!file) {
        printf("Error opening file %s for writing.\n", csv_filename);
        return;
    }

    fprintf(file, "Epoch");
    for (int i = 0; i <= optimizer->input_count; i++) {
        fprintf(file, ",w%d", i);
    }
    fprintf(file, "\n");

    FILE* file2 = fopen(csv_filename2, "w");
    if (!file2) {
        printf("Error opening file %s for writing.\n", csv_filename2);
        return;
    }

    fprintf(file2, "Epoch,Time,Loss\n");
    double cumulative_time = 0.0;

    
    double initial_loss = 0.0;
    for (int i = 0; i < dataset->num_samples; i++) {
        double prediction = predict(optimizer, dataset->data[i]);
        double true_label = dataset->labels[i];
        double error = true_label - prediction;
        initial_loss += error * error;
    }
    initial_loss /= dataset->num_samples;
    fprintf(file2, "0,0.000000,%f\n", initial_loss);
    log_weights_to_csv(file, 0, optimizer->weights, optimizer->input_count + 1);

    printf("Epoch 0: Loss = %f, Cumulative Time = 0.000000 seconds\n", initial_loss);

    for (int epoch = 1; epoch <= iterations; epoch++) {
        clock_t start_time = clock();

        double total_error = 0.0;

        for (int i = 0; i < dataset->num_samples; i++) {
            int j = rand() % dataset->num_samples;

            double* temp_data = dataset->data[i];
            dataset->data[i] = dataset->data[j];
            dataset->data[j] = temp_data;

            double temp_label = dataset->labels[i];
            dataset->labels[i] = dataset->labels[j];
            dataset->labels[j] = temp_label;
        }

        for (int i = 0; i < dataset->num_samples; i++) {
            t++;

            double prediction = predict(optimizer, dataset->data[i]);
            double true_label = dataset->labels[i];

            double error = true_label - prediction;
            total_error += error * error;

            double gradient = error * (1 - prediction * prediction);  

            for (int j = 0; j < optimizer->input_count + 1; j++) {
                double grad = (j == 0) ? gradient : gradient * dataset->data[i][j - 1];

                optimizer->m[j] = BETA1 * optimizer->m[j] + (1 - BETA1) * grad;
                optimizer->v[j] = BETA2 * optimizer->v[j] + (1 - BETA2) * grad * grad;

                double m_hat = optimizer->m[j] / (1 - pow(BETA1, t));
                double v_hat = optimizer->v[j] / (1 - pow(BETA2, t));

                optimizer->weights[j] += LEARNING_RATE * m_hat / (sqrt(v_hat) + EPSILON);
            }
        }

        clock_t end_time = clock();
        double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        cumulative_time += elapsed_time;

        double avg_loss = total_error / dataset->num_samples;

        fprintf(file2, "%d,%f,%f\n", epoch, cumulative_time, avg_loss);
        log_weights_to_csv(file, epoch, optimizer->weights, optimizer->input_count + 1);

        if (epoch % 10 == 0 || epoch == iterations) {
            printf("Epoch %d: Loss = %f, Cumulative Time = %f seconds\n", epoch, avg_loss, cumulative_time);
        }
    }

    fclose(file);
    fclose(file2);
}


// Adam için modeli test eder
void test_model(AdamOptimizer* optimizer, Dataset* test_dataset) {
    double total_accuracy = 0.0;
    double total_error = 0.0;

    for (int i = 0; i < test_dataset->num_samples; i++) {
        double prediction = predict(optimizer, test_dataset->data[i]);
        double true_label = test_dataset->labels[i];

        double error = true_label - prediction;
        total_error += error * error;

        total_accuracy += (fabs(round(prediction) - true_label) < 0.5) ? 1.0 : 0.0;
    }

    printf("\nTest Results:\n");
    printf("Test Error: %f\n", total_error / test_dataset->num_samples);
    printf("Test Accuracy: %f%%\n", 
           (total_accuracy / test_dataset->num_samples) * 100);
}

// SGD için modeli test eder
void test_model_sgd(SGDOptimizer* optimizer, Dataset* test_dataset) {
    double total_accuracy = 0.0;
    double total_error = 0.0;

    for (int i = 0; i < test_dataset->num_samples; i++) {
        double prediction = predict_sgd(optimizer, test_dataset->data[i]);
        double true_label = test_dataset->labels[i];

        double error = true_label - prediction;
        total_error += error * error;

        total_accuracy += (fabs(round(prediction) - true_label) < 0.5) ? 1.0 : 0.0;
    }

    printf("\nTest Results:\n");
    printf("Test Error: %f\n", total_error / test_dataset->num_samples);
    printf("Test Accuracy: %f%%\n", 
           (total_accuracy / test_dataset->num_samples) * 100);
}



int main() {
    Dataset* train_dataset;
    Dataset* test_dataset;

    read_and_split_csv("mnist.csv", &train_dataset, &test_dataset);

    if (!train_dataset || !test_dataset) {
        printf("Error\n");
        return 1;
    }

    SGDOptimizer* optimizer = create_sgd_optimizer(INPUT_SIZE, LEARNING_RATE);

    printf("SGD training:\n");
    train_sgd(optimizer, train_dataset, "weights_log_sgd.csv", "training_log.csv", MAX_EPOCHS);

    test_model_sgd(optimizer, test_dataset);

    free_sgd_optimizer(optimizer);
    AdamOptimizer* adam_optimizer = create_adam_optimizer(INPUT_SIZE);

    printf("Adam training:\n");
    train_adam(adam_optimizer, train_dataset, "weights_log_adam.csv", "training_log_adam.csv", MAX_EPOCHS);

    test_model(adam_optimizer, test_dataset);

    free_optimizer(adam_optimizer);

    printf("\nGD training:\n");
    GDOptimizer* gd_optimizer = create_gd_optimizer(INPUT_SIZE, LEARNING_RATE);
    train_gd(gd_optimizer, train_dataset, "weights_log_gd.csv", "training_log_gd.csv", MAX_EPOCHS);

    test_model_gd(gd_optimizer, test_dataset);

    free_gd_optimizer(gd_optimizer);


    free_dataset(train_dataset);
    free_dataset(test_dataset);

    return 0;
}
