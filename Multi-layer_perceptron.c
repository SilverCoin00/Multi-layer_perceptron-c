#pragma once
#include "D:\Data\code_doc\AI_model_building\Multi-layer_perceptron\Core.h"

typedef struct {
    int nodes;
    char* activation;
} Dense;
typedef struct {
    float drop;
} Dropout;
typedef struct {
    Dense* dense;
    Dropout* dropout;
} Keras_layer;
typedef struct Optimizer {
    int type;
    float learning_rate;
    float momentum;
    int nesterov;
    float beta_1, beta_2;
    float rho;
    float epsilon;
    float init_accumulator_grad;
    Weights** pre_velocity;
    Weights** accumulator_grad;
} Optimizer;
typedef struct Model_Compiler {
    Optimizer* optimize;
    int loss_type, metrics_type;
} Model_Compiler;
typedef struct {
    int monitor;
    float baseline;
    int patience;
    float min_delta;
    int restore_best_weights;
    float last_monitor_val;
    float best_monitor_val;
} Early_Stopping;
typedef struct MLP {
    Dataset_2* train_data, * val_data;
    Weights** w;
    float*** z;           // = X(T).W + B(T), size == size(a)
    float*** a;           // = f(z), sizeof(num_layers x samples x num_units)
    int* activation;
    int* num_units;
    int num_layers, cur_batch_size;
    Model_Compiler* compiler;
    Weights** deriv;
    Dropout** dropout;
    int* drop_layer;
} MLP;

static int activation_encode(char* activation) {
    if (!strcmp(activation, "none") || !strcmp(activation, "linear")) return 0;
    else if (!strcmp(activation, "relu")) return 1;
    else if (!strcmp(activation, "sigmoid")) return 2;
    else if (!strcmp(activation, "tanh")) return 3;
    else if (!strcmp(activation, "softmax")) return 4;
    else if (!strcmp(activation, "softplus")) return 5;
    return -1;
}
static float** activation_func(float** z, int samples, int num_units, int activate_type) {
    int i, j;
    float** a = new_matrix(samples, num_units);
    if (activate_type == 0) {
        for (i = 0; i < samples; i++) {
            for (j = 0; j < num_units; j++) a[i][j] = z[i][j];
        }
    } else if (activate_type == 1) {
        for (i = 0; i < samples; i++) {
            for (j = 0; j < num_units; j++) if (z[i][j] > 0) a[i][j] = z[i][j];
        }
    } else if (activate_type == 2) {
        for (i = 0; i < samples; i++) {
            for (j = 0; j < num_units; j++) a[i][j] = 1 / (1 + exp(-z[i][j]));
        }
    } else if (activate_type == 3) {
        for (i = 0; i < samples; i++) {
            for (j = 0; j < num_units; j++) a[i][j] = 2 / (1 + exp(-2* z[i][j])) - 1;
        }
    } else if (activate_type == 4) {
        float sum, max;
        for (i = 0; i < samples; i++) {
            sum = 0;
            max = z[i][0];
            for (j = 1; j < num_units; j++) if (z[i][j] > max) max = z[i][j];
            for (j = 0; j < num_units; j++) {
                a[i][j] = exp(z[i][j] - max);
                sum += a[i][j];
            }
            for (j = 0; j < num_units; j++) a[i][j] /= sum;
        }
    } else if (activate_type == 5) {
        for (i = 0; i < samples; i++) {
            for (j = 0; j < num_units; j++) a[i][j] = log(1 + exp(z[i][j]));
        }
    }
    return a;
}
static float** activation_derivative(float** a, float** z, int samples, int num_units, int activate_type) {  // da|dz = f'(z)
    int i, j;
    float** deriv = new_matrix(samples, num_units);
    if (activate_type == 0) {
        for (i = 0; i < samples; i++)
            for (j = 0; j < num_units; j++) deriv[i][j] = 1;
    } else if (activate_type == 1) {
        for (i = 0; i < samples; i++)
            for (j = 0; j < num_units; j++) deriv[i][j] = (z[i][j] > 0.0f) ? 1.0f : 0.0f;
    } else if (activate_type == 2) {
        for (i = 0; i < samples; i++)
            for (j = 0; j < num_units; j++) deriv[i][j] = a[i][j]* (1 - a[i][j]);
    } else if (activate_type == 3) {
        for (i = 0; i < samples; i++)
            for (j = 0; j < num_units; j++) deriv[i][j] = 1 - a[i][j]* a[i][j];
    } else if (activate_type == 5) {
        for (i = 0; i < samples; i++)
            for (j = 0; j < num_units; j++) deriv[i][j] = 1 / (1 + exp(-z[i][j]));
    }
    return deriv;
}
static void free_activated_data(MLP* model, int layer, int samples) {
    if (!model->a[layer] || !model->z[layer]) return ;
    for (int i = 0; i < samples; i++) {
        free(model->a[layer][i]);
        free(model->z[layer][i]);
    }
    free(model->a[layer]);
    free(model->z[layer]);
}
static int read_keras_layers(Keras_layer** layers, int* dense, int* activation, Dropout** dropout, int* drop_layer) {
    for (int i = 0, j = 0, k = 0; layers[i]; i++) {
        if (layers[i]->dense) {
            dense[j] = layers[i]->dense->nodes;
            if (dense[j] <= 0) {
                printf("Error: Inappropriate layer num of nodes !!");
                return 0;
            }
            activation[j++] = activation_encode(layers[i]->dense->activation);
        } else if (layers[i]->dropout) {
            dropout[k] = (Dropout*)malloc(sizeof(Dropout));
            if (layers[i]->dropout->drop >= 1 || layers[i]->dropout->drop < 0) {
                printf("Error: Inappropriate dropout ratio !!");
                return 0;
            }
            dropout[k++]->drop = layers[i]->dropout->drop;
            drop_layer[k] = j;
        }
    }
    return 1;
}
void free_mlp_model(MLP* model) {
    int i;
    free_dataset2(model->train_data);
    free_dataset2(model->val_data);
    if (model->w) {
        for (i = 0; i < model->num_layers; i++) free_weights(model->w[i]);
        free(model->w);
    }
    if (model->deriv) {
        for (i = 0; i < model->num_layers; i++) free_weights(model->deriv[i]);
        free(model->deriv);
    }
    if (model->a && model->z) {
        for (i = 0; i < model->num_layers; i++) free_activated_data(model, i, model->cur_batch_size);
        free(model->a);
        free(model->z);
    }
    if (model->compiler) {
        if (model->compiler->optimize) {
            if (model->compiler->optimize->pre_velocity) {
                for (i = 0; i < model->num_layers; i++) free_weights(model->compiler->optimize->pre_velocity[i]);
                free(model->compiler->optimize->pre_velocity);
            }
            if (model->compiler->optimize->accumulator_grad) {
                for (i = 0; i < model->num_layers; i++) free_weights(model->compiler->optimize->accumulator_grad[i]);
                free(model->compiler->optimize->accumulator_grad);
            }
            free(model->compiler->optimize);
        }
        free(model->compiler);
    }
    for (i = 0; i < model->drop_layer[0]; i++) free(model->dropout[i]);
    if (model->drop_layer[0]) free(model->dropout);
    free(model->drop_layer);
    free(model->activation);
    free(model->num_units);
    free(model);
}
MLP* new_model(int* input_shape, Keras_layer** layers) {
    int i, num_layers = 0, dropout = 0;
    for (i = 0; layers[i]; i++) {
        if (layers[i]->dense) num_layers++;
        else if (layers[i]->dropout) dropout++;
    }
    if (num_layers < 2) {
        printf("Warning: MLP dont support lower than 2 layers !!");
        return NULL;
    }
    MLP* model = (MLP*)calloc(1, sizeof(MLP));
    if (dropout) model->dropout = (Dropout**)malloc(dropout* sizeof(Dropout*));
    model->drop_layer = (int*)malloc((dropout + 1)* sizeof(int));
    model->drop_layer[0] = dropout;
    model->activation = (int*)malloc(num_layers* sizeof(int));
    model->num_units = (int*)malloc(num_layers* sizeof(int));
    if (!read_keras_layers(layers, model->num_units, model->activation, model->dropout, model->drop_layer)) {
        free_mlp_model(model);
        return NULL;
    }

    model->w = (Weights**)malloc(num_layers* sizeof(Weights*));
    model->w[0] = init_weights(input_shape[1], model->num_units[0], time(NULL));
    for (i = 1; i < num_layers; i++) model->w[i] = init_weights(model->num_units[i - 1], model->num_units[i], time(NULL));
    model->z = (float***)calloc(num_layers, sizeof(float**));
    model->a = (float***)calloc(num_layers, sizeof(float**));
    
    model->num_layers = num_layers;
    model->cur_batch_size = input_shape[0];
    return model;
}
Optimizer* new_optimizer(char* optimizer, float learning_rate, float momentum, int nesterov, 
                float beta_1, float beta_2, float rho, float epsilon, float Grad_0) {
    Optimizer* newo = (Optimizer*)calloc(1, sizeof(Optimizer));
    newo->momentum = 0;
    newo->nesterov = 0;
    newo->init_accumulator_grad = 0;
    if (!strcmp(optimizer, "SGD")) {
        newo->type = 0;
        newo->learning_rate = learning_rate > 0 ? learning_rate : 0.01f;
        newo->momentum = momentum > 0 ? momentum : 0;
        newo->nesterov = !!nesterov;
    } else if (!strcmp(optimizer, "Adagrad")) {
        newo->type = 1;
        newo->learning_rate = learning_rate > 0 ? learning_rate : 0.001f;
        newo->init_accumulator_grad = Grad_0 > 0 ? Grad_0 : 0.1f;
        newo->epsilon = epsilon > 0 ? epsilon : 1e-7f;
    } else if (!strcmp(optimizer, "RMSProp")) {
        newo->type = 2;
        newo->learning_rate = learning_rate > 0 ? learning_rate : 0.001f;
        newo->rho = rho > 0 ? rho : 0.9f;
        newo->epsilon = epsilon > 0 ? epsilon : 1e-7f;
        newo->momentum = momentum > 0 ? momentum : 0;
    } else if (!strcmp(optimizer, "Adadelta")) {
        newo->type = 3;
        newo->learning_rate = learning_rate > 0 ? learning_rate : 0.001f;
        newo->rho = rho > 0 ? rho : 0.95f;
        newo->epsilon = epsilon > 0 ? epsilon : 1e-7f;
    } else if (!strcmp(optimizer, "Adam")) {
        newo->type = 4;
        newo->learning_rate = learning_rate > 0 ? learning_rate : 0.001f;
        newo->epsilon = epsilon > 0 ? epsilon : 1e-7f;
        newo->beta_1 = beta_1 > 0 ? beta_1 : 0.9f;
        newo->beta_2 = beta_2 > 0 ? beta_2 : 0.999f;
    } else {
        free(newo);
        printf("Warning: Unreconized optimizer type !!");
        return NULL;
    }
    return newo;
}
const char* Monitors[] = {"loss", "accuracy", "val_loss", "val_accuracy"};
Early_Stopping* new_earlystopping(char* monitor, float min_delta, int patience, float baseline, int restore_best_weights) {
    Early_Stopping* new_es = (Early_Stopping*)malloc(sizeof(Early_Stopping));
    int i;
    for (i = 0; i < 4; i++) {
        if (!strcmp(monitor, Monitors[i])) {
            new_es->monitor = i;
            if (i % 2 == 0) new_es->last_monitor_val = 1.0f / 0.0f;
            else new_es->last_monitor_val = 0.0f;
            new_es->best_monitor_val = new_es->last_monitor_val;
            break;
        }
    }
    if (i == 4) {
        printf("Error: Unreconized monitor type !!");
        free(new_es);
        return NULL;
    }
    new_es->patience = patience > 0 ? patience : 0;
    new_es->min_delta = min_delta > 0 ? min_delta : 0;
    new_es->baseline = baseline;
    new_es->restore_best_weights = !!restore_best_weights;
    return new_es;
}
int early_stopping_check(Early_Stopping* es, float monitor_val) {
    if (es->monitor % 2 == 0) {
        if (monitor_val > es->last_monitor_val - es->min_delta) return 0;  // not improve
        else {
            if (monitor_val < es->best_monitor_val) {
                es->best_monitor_val = monitor_val;
                return 2;
            } else return 1;
        }
    } else {
        if (monitor_val < es->last_monitor_val + es->min_delta) return 0;
        else {
            if (monitor_val > es->best_monitor_val) {
                es->best_monitor_val = monitor_val;
                return 2;
            } else return 1;
        }
    }
}
const char *Metrics[] = {"accuracy", "mse", "mae", "binary_accuracy", "categorical_accuracy"};
void model_compile(MLP* model, Optimizer* optimize, char* loss, char* metrics) {
    model->compiler = (Model_Compiler*)malloc(sizeof(Model_Compiler));
    model->compiler->optimize = optimize;
    int i;
    model->compiler->optimize->pre_velocity = (Weights**)calloc(model->num_layers, sizeof(Weights*));
    for (i = 1; i < model->num_layers; i++)
        model->compiler->optimize->pre_velocity[i] = init_weights(model->num_units[i - 1], model->num_units[i], 0);
    model->compiler->optimize->accumulator_grad = (Weights**)calloc(model->num_layers, sizeof(Weights*));
    for (i = 1; i < model->num_layers; i++)
        model->compiler->optimize->accumulator_grad[i] = fill_weights(model->num_units[i - 1], model->num_units[i], model->compiler->optimize->init_accumulator_grad);
    if (!strcmp(loss, "mse")) model->compiler->loss_type = 1;
    else if (!strcmp(loss, "binary_crossentropy")) model->compiler->loss_type = 2;
    else if (!strcmp(loss, "categorical_crossentropy")) model->compiler->loss_type = 3;
    for (i = 0; i < 5; i++) {
        if (!strcmp(Metrics[i], metrics)) {
            model->compiler->metrics_type = i;
            break;
        }
    }
}
static float loss_func(int loss_type, float** y_pred, float** y_true, int classes, int samples) {
    int i, j;
    float loss = 0;
    if (loss_type == 1) {
        for (i = 0; i < samples; i++)
            for (j = 0; j < classes; j++) loss += (y_pred[i][j] - y_true[i][j])* (y_pred[i][j] - y_true[i][j]);
        loss /= classes;
    } else if (loss_type == 2) {
        float pred;
        for (i = 0; i < samples; i++) {
            pred = y_pred[i][0];
            pred = pred > 1e-8 ? pred : 1e-8;
            pred = pred < 1 - 1e-8 ? pred : 1 - 1e-8;
            loss += - (y_true[i][0]* logf(pred) + (1 - y_true[i][0])* logf(1 - pred));
        }
    } else if (loss_type == 3) {
        for (i = 0; i < samples; i++)
            for (j = 0; j < classes; j++) if (y_true[i][j] == 1.0) loss -= logf(y_pred[i][j] + 1e-10);
    }
    return loss / samples;
}
static float metrics_func(int metrics_type, float** y_pred, float** y_true, int classes, int samples) {
    int i, j;
    float metric = 0;
    if (metrics_type == 0) {
        if (classes == 1) metrics_type = 3;
        else metrics_type = 4;
    } else if (metrics_type == 1) {
        for (i = 0; i < samples; i++)
            for (j = 0; j < classes; j++) metric += (y_pred[i][j] - y_true[i][j])* (y_pred[i][j] - y_true[i][j]);
        metric /= classes;
    } else if (metrics_type == 2) {
        for (i = 0; i < samples; i++)
            for (j = 0; j < classes; j++) metric += fabs(y_pred[i][j] - y_true[i][j]);
        metric /= classes;
    } else if (metrics_type == 3) {
        for (i = 0; i < samples; i++)
            metric += (y_pred[i][0] > 0.5 ? 1 : 0) == y_true[i][0];
    } else if (metrics_type == 4) {
        int max;
        for (i = 0; i < samples; i++) {
            max = 0;
            for (j = 1; j < classes; j++) if (y_pred[i][j] > y_pred[i][max]) max = j;
            metric += y_true[i][max];
        }
    }
    return metric / samples;
}
void drop_neural(float** a, int samples, int* drop_i, int* units, float drop_ratio) {
    if (!(*units) || !drop_ratio) return ;
    int num_drop = round(drop_ratio* *units);
    shuffle_index(drop_i, *units, time(NULL));
    int i, j;
    for (i = 1; i <= num_drop; i++) {
        for (j = 0; j < samples; j++) a[j][drop_i[*units - i]] = 0;
    }
    for ( ; i <= *units; i++) {
        for (j = 0; j < samples; j++) a[j][drop_i[*units - i]] /= (1 - drop_ratio);
    }
    *units -= num_drop;
}
void predict(MLP* model, Dataset_2* data, float** y_pred, int is_training) {
    if (data->features != model->w[0]->row) {
        printf("Error: Input data size is inappropriate !!");
        return ;
    }
    int i, j, k, l = 1, * drop_i;
    if (is_training && model->drop_layer[0]) {
        k = data->features;
        drop_i = (int*)malloc(k* sizeof(int));
        for (i = 0; i < k; i++) drop_i[i] = i;
        for ( ; l <= model->drop_layer[0] && model->drop_layer[l] == 0; l++)
            drop_neural(data->x, data->samples, drop_i, &k, model->dropout[l - 1]->drop);
        free(drop_i);
    }
    free_activated_data(model, 0, model->cur_batch_size);
    model->z[0] = matrix_multiply(data->x, model->w[0]->weights, data->samples, data->features, model->num_units[0]);
    for (i = 0; i < data->samples; i++) {
        for (j = 0; j < model->num_units[0]; j++) model->z[0][i][j] += model->w[0]->bias[j];
    }
    model->a[0] = activation_func(model->z[0], data->samples, model->num_units[0], model->activation[0]);
    for (i = 1; i < model->num_layers; i++) {
        if (is_training && model->drop_layer[0]) {
            k = model->num_units[i - 1];
            drop_i = (int*)malloc(k* sizeof(int));
            for (j = 0; j < k; j++) drop_i[j] = j;
            for ( ; l <= model->drop_layer[0] && model->drop_layer[l] == i; l++)
                drop_neural(model->a[i - 1], data->samples, drop_i, &k, model->dropout[l - 1]->drop);
            free(drop_i);
        }
        free_activated_data(model, i, model->cur_batch_size);
        model->z[i] = matrix_multiply(model->a[i - 1], model->w[i]->weights, data->samples, model->num_units[i - 1], model->num_units[i]);
        for (k = 0; k < data->samples; k++) {
            for (j = 0; j < model->num_units[i]; j++) model->z[i][k][j] += model->w[i]->bias[j];
        }
        model->a[i] = activation_func(model->z[i], data->samples, model->num_units[i], model->activation[i]);
    }
    for (i = 0, k = model->num_layers - 1; i < data->samples; i++) {
        for (j = 0; j < model->num_units[k]; j++) y_pred[i][j] = model->a[k][i][j];
    }
    model->cur_batch_size = data->samples;
}
void backpropagation(MLP* model, Dataset_2* data) {
    if (!model->deriv) model->deriv = (Weights**)calloc(model->num_layers, sizeof(Weights*));
    int i = model->num_layers - 1, j, k, l;
    if (model->deriv[i]) free_weights(model->deriv[i]);
    model->cur_batch_size = data->samples;
    float** cur_delta = new_matrix(model->cur_batch_size, model->num_units[i]);
    float** back_delta;
    float** a_deriv = NULL;
    float nesterov = (float) model->compiler->optimize->nesterov, moment = model->compiler->optimize->momentum;
    Weights* tempw;

    for (j = 0; j < data->samples; j++) {
        for (k = 0; k < model->num_units[i]; k++) cur_delta[j][k] = (model->a[i][j][k] - data->y[j][k]) / data->samples;
    }
    if (model->compiler->loss_type == 1 && model->activation[i] != 0) {
        a_deriv = activation_derivative(model->a[i], model->z[i], model->cur_batch_size, model->num_units[i], model->activation[i]);
        for (j = 0; j < model->cur_batch_size; j++) {
            for (k = 0; k < model->num_units[i]; k++) cur_delta[j][k] *= a_deriv[j][k];
        }
        free_matrix(a_deriv, model->cur_batch_size);
    }
    model->deriv[i] = weights_derivative(model->a[i - 1], model->cur_batch_size, model->num_units[i - 1], cur_delta, model->num_units[i]);
    for (i--; i > 0; i--) {
        back_delta = cur_delta;
        a_deriv = activation_derivative(model->a[i], model->z[i], model->cur_batch_size, model->num_units[i], model->activation[i]);
        tempw = copy_weights(model->w[i + 1]);
        for (k = 0; k < tempw->col; k++) {
            for (j = 0; j < tempw->row; j++)
                tempw->weights[j][k] -= nesterov* moment* model->compiler->optimize->pre_velocity[i + 1]->weights[j][k];
            tempw->bias[k] -= nesterov* moment* model->compiler->optimize->pre_velocity[i + 1]->bias[k];
        }
        cur_delta = cal_delta(tempw, back_delta, a_deriv, model->cur_batch_size);
        if (model->deriv[i]) free_weights(model->deriv[i]);
        model->deriv[i] = weights_derivative(model->a[i - 1], model->cur_batch_size, model->num_units[i - 1], cur_delta, model->num_units[i]);
        free_matrix(a_deriv, model->cur_batch_size);
        free_matrix(back_delta, model->cur_batch_size);
        free_weights(tempw);
    }
    if (i == 0) {
        back_delta = cur_delta;
        a_deriv = activation_derivative(model->a[0], model->z[0], model->cur_batch_size, model->num_units[0], model->activation[0]);
        tempw = copy_weights(model->w[1]);
        for (k = 0; k < tempw->col; k++) {
            for (j = 0; j < tempw->row; j++)
                tempw->weights[j][k] -= nesterov* moment* model->compiler->optimize->pre_velocity[1]->weights[j][k];
            tempw->bias[k] -= nesterov* moment* model->compiler->optimize->pre_velocity[1]->bias[k];
        }
        cur_delta = cal_delta(tempw, cur_delta, a_deriv, model->cur_batch_size);
        if (model->deriv[0]) free_weights(model->deriv[0]);
        model->deriv[0] = weights_derivative(data->x, model->cur_batch_size, data->features, cur_delta, model->num_units[0]);
        free_matrix(a_deriv, model->cur_batch_size);
        free_matrix(back_delta, model->cur_batch_size);
        free_matrix(cur_delta, model->cur_batch_size);
        free_weights(tempw);
    } else /* only for supporting one layer perceptron */ ;
}
void update_model_weights(MLP* model, int times) {
    int i;
    if (model->compiler->optimize->type == 0) {
        for (i = 0; i < model->num_layers; i++)
            grad_descent(model->w[i], model->deriv[i], model->compiler->optimize->learning_rate, 
                model->compiler->optimize->pre_velocity[i], model->compiler->optimize->momentum);
    } else if (model->compiler->optimize->type == 1) {
        for (i = 0; i < model->num_layers; i++) 
            adaptive_grad_descent(model->w[i], model->deriv[i], model->compiler->optimize->learning_rate, 
                model->compiler->optimize->accumulator_grad[i], model->compiler->optimize->epsilon);
    } else if (model->compiler->optimize->type == 2) {
        for (i = 0; i < model->num_layers; i++) 
            root_mean_square_propagation(model->w[i], model->deriv[i], model->compiler->optimize->learning_rate, 
                model->compiler->optimize->accumulator_grad[i], model->compiler->optimize->rho, 
                model->compiler->optimize->epsilon, model->compiler->optimize->pre_velocity[i], 
                model->compiler->optimize->momentum);
    } else if (model->compiler->optimize->type == 3) {
        for (i = 0; i < model->num_layers; i++) 
            adaptive_delta_grad(model->w[i], model->deriv[i], model->compiler->optimize->learning_rate, 
                model->compiler->optimize->accumulator_grad[i], model->compiler->optimize->rho, 
                model->compiler->optimize->epsilon, model->compiler->optimize->pre_velocity[i]);
    } else if (model->compiler->optimize->type == 4) {
        for (i = 0; i < model->num_layers; i++) 
            adaptive_moment_estimation(model->w[i], model->deriv[i], model->compiler->optimize->learning_rate, 
                model->compiler->optimize->accumulator_grad[i], model->compiler->optimize->beta_1, 
                model->compiler->optimize->beta_2, model->compiler->optimize->epsilon, 
                model->compiler->optimize->pre_velocity[i], times);
    }
}
void model_fit(MLP* model, Dataset_2* data, int epochs, int batch_size, float validation_split, void* call_backs) {
    int i = model->w[0]->row == data->features, j = model->num_units[model->num_layers - 1] == data->y_types;
    if (!(i && j)) {
        printf("Warning: Model and data %s%s%s sizes are incompatible !!", i ? "" : "inputs", i^j ? "" : ", ", j ? "" : "outputs");
        return ;
    }
    float** y_pred, ** y_pval;
    float loss, metrics, val_loss, val_metrics;
    free_dataset2(model->train_data);
    free_dataset2(model->val_data);
    if (validation_split == 0) model->train_data = data;
    else {
        if (validation_split < 0 || validation_split >= 1) validation_split = 0.2;
        model->train_data = (Dataset_2*)malloc(sizeof(Dataset_2));
        model->val_data = (Dataset_2*)malloc(sizeof(Dataset_2));
        train_test_split_ds2(data, model->train_data, model->val_data, validation_split, data->y_types);
        y_pval = new_matrix(model->val_data->samples, data->y_types);
    }
    Early_Stopping* estop;
    Weights** restore_w;
    int stop_threshold = 0, best_epoch;
    float* monitor;
    if (call_backs) {
        estop = (Early_Stopping*) call_backs;
        if (estop->restore_best_weights) {
            best_epoch = 0;
            restore_w = (Weights**)malloc(model->num_layers* sizeof(Weights*));
            for (i = 0, j = model->num_layers; i < j; i++) restore_w[i] = copy_weights(model->w[i]);
        }
        if (estop->monitor == 0) monitor = &loss;
        else if (estop->monitor == 1) monitor = &metrics;
        else if (estop->monitor == 2) monitor = &val_loss;
        else if (estop->monitor == 3) monitor = &val_metrics;
    }
    model->compiler->optimize->pre_velocity[0] = init_weights(data->features, model->num_units[0], 0);
    model->compiler->optimize->accumulator_grad[0] = fill_weights(data->features, model->num_units[0], model->compiler->optimize->init_accumulator_grad);
    
    int* random_i = (int*)malloc(model->train_data->samples* sizeof(int)), loop, k, h;
    for (i = 0; i < model->train_data->samples; i++) random_i[i] = i;
    Dataset_2* batch;

    if (batch_size <= 0 || batch_size >= model->train_data->samples) batch_size = model->train_data->samples;
	else shuffle_index(random_i, model->train_data->samples, i);
    model->cur_batch_size = batch_size;
	loop = model->train_data->samples / batch_size;
    y_pred = new_matrix(batch_size, data->y_types);

    for (i = 1; i <= epochs; i++) {
        loss = 0, metrics = 0, val_loss = 0, val_metrics = 0;
        shuffle_index(random_i, model->train_data->samples, i);
        for (j = 0; j < loop; j++) {
            batch = dataset2_samples_order_copy(model->train_data, random_i, j* batch_size, (j + 1)* batch_size);
            predict(model, batch, y_pred, 1);
            loss += loss_func(model->compiler->loss_type, y_pred, batch->y, batch->y_types, batch->samples);
            metrics += metrics_func(model->compiler->metrics_type, y_pred, batch->y, batch->y_types, batch->samples);
            backpropagation(model, batch);
            update_model_weights(model, i*(j + 1));
            free_dataset2(batch);
        }
        h = 0;
        if (batch_size* loop < model->train_data->samples) {
            h++;
            batch = dataset2_samples_order_copy(model->train_data, random_i, j* batch_size, model->train_data->samples);
            predict(model, batch, y_pred, 1);
            loss += loss_func(model->compiler->loss_type, y_pred, batch->y, batch->y_types, batch->samples);
            metrics += metrics_func(model->compiler->metrics_type, y_pred, batch->y, batch->y_types, batch->samples);
            backpropagation(model, batch);
            update_model_weights(model, i*(j + 1));
            free_dataset2(batch);
        }
        loss /= (loop + h), metrics /= (loop + h);
        printf("Epoch %d/%d\n%d/%d [==============================] - loss: %.4f - %s: %.4f", 
            i, epochs, loop + h, loop + h, loss, Metrics[model->compiler->metrics_type], metrics);
        if (validation_split) {
            predict(model, model->val_data, y_pval, 0);
            val_loss = loss_func(model->compiler->loss_type, y_pval, model->val_data->y, data->y_types, model->val_data->samples);
            val_metrics = metrics_func(model->compiler->metrics_type, y_pval, model->val_data->y, data->y_types, model->val_data->samples);
            printf(" - val_loss: %.4f - val_%s: %.4f", val_loss, Metrics[model->compiler->metrics_type], val_metrics);
        }
        if (call_backs) {
            h = early_stopping_check(estop, *monitor);
            if(!h) stop_threshold++;
            else {
                stop_threshold = 0;
                if (h == 2 && estop->restore_best_weights) {
                    for (k = 0, j = model->num_layers; k < j; k++) {
                        free_weights(restore_w[k]);
                        restore_w[k] = copy_weights(model->w[k]);
                    }
                    best_epoch = i;
                }
            }
            if (stop_threshold >= estop->patience) {
                printf("\nEpoch %d: early stopping\n", i);
                break;
            }
            estop->last_monitor_val = *monitor;
        }
        printf("\n");
    }
    if (call_backs && estop->restore_best_weights) {
        printf("Restoring model weights from the end of the best epoch: %d\n", best_epoch);
        for (k = 0, j = model->num_layers; k < j; k++) {
            free_weights(model->w[k]);
            model->w[k] = copy_weights(restore_w[k]);
            free_weights(restore_w[k]);
        }
        free(restore_w);
    }
    free(random_i);
    free_matrix(y_pred, batch_size);
    if (validation_split) free_matrix(y_pval, model->val_data->samples);
}
void model_evaluate(MLP* model, Dataset_2* data) {
    float** y_pred = new_matrix(data->samples, data->y_types);
    predict(model, data, y_pred, 0);
    float loss = loss_func(model->compiler->loss_type, y_pred, data->y, data->y_types, data->samples);
    float metrics = metrics_func(model->compiler->metrics_type, y_pred, data->y, data->y_types, data->samples);
    printf("Evaluate on test data:\n%d/%d [==============================] - loss: %.4f - %s: %.4f\n", 
            data->samples, data->samples, loss, Metrics[model->compiler->metrics_type], metrics);
    free_matrix(y_pred, data->samples);
}