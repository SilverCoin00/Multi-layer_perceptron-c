#pragma once
#include "Core.h"

typedef struct Weights {
    float** weights;     // num_front_units x num_back_units
    float* bias;         // num_back_units x 1
    int row, col;
} Weights;

Weights* init_weights(int row, int col, int random_state) {
    Weights* neww = (Weights*)malloc(sizeof(Weights));
    int i, j;
    srand(random_state);
    neww->weights = (float**)malloc(row* sizeof(float*));
    for (i = 0; i < row; i++) {
        neww->weights[i] = (float*)calloc(col, sizeof(float));
        if (random_state) for (j = 0; j < col; j++) neww->weights[i][j] = ((float)rand() / RAND_MAX)* 0.2 - 0.1;
    }
    neww->bias = (float*)calloc(col, sizeof(float));
    if (random_state) for (i = 0; i < col; i++) neww->bias[i] = ((float)rand() / RAND_MAX)* 0.2 - 0.1;
    neww->row = row;
    neww->col = col;
    return neww;
}
Weights* fill_weights(int row, int col, float fill_value) {
    Weights* neww = (Weights*)malloc(sizeof(Weights));
    int i, j;
    neww->weights = (float**)malloc(row* sizeof(float*));
    for (i = 0; i < row; i++) {
        neww->weights[i] = (float*)malloc(col* sizeof(float));
        for (j = 0; j < col; j++) neww->weights[i][j] = fill_value;
    }
    neww->bias = (float*)malloc(col* sizeof(float));
    for (i = 0; i < col; i++) neww->bias[i] = fill_value;
    neww->row = row;
    neww->col = col;
    return neww;
}
Weights* copy_weights(const Weights* sauce) {
    Weights* neww = (Weights*)malloc(sizeof(Weights));
    int i, j;
    neww->weights = (float**)malloc(sauce->row* sizeof(float*));
    for (i = 0; i < sauce->row; i++) {
        neww->weights[i] = (float*)calloc(sauce->col, sizeof(float));
        for (j = 0; j < sauce->col; j++) neww->weights[i][j] = sauce->weights[i][j];
    }
    neww->bias = (float*)calloc(sauce->col, sizeof(float));
    for (i = 0; i < sauce->col; i++) neww->bias[i] = sauce->bias[i];
    neww->row = sauce->row;
    neww->col = sauce->col;
    return neww;
}
float** cal_delta(Weights* w, float** back_delta, float** a_deriv, int samples) {  // &e(l) = (&e(l+1). w(l+1)(T)) @ f'(z(l))
    float** w_T = transpose_matrix(w->weights, w->row, w->col);
    float** dL_da = matrix_multiply(back_delta, w_T, samples, w->col, w->row);
    free_matrix(w_T, w->col);
    float** delta = element_wise_multiply(dL_da, a_deriv, samples, w->row);
    free_matrix(dL_da, samples);
    return delta;
}
Weights* weights_derivative(float** a, int samples, int front_units, float** delta, int units) {  // = 1/n. a(l-1)(T).&e(l)
    Weights* deriv = (Weights*)malloc(sizeof(Weights));
    deriv->row = front_units, deriv->col = units;
    float** a_T = transpose_matrix(a, samples, front_units);
    deriv->weights = matrix_multiply(a_T, delta, front_units, samples, units);
    deriv->bias = (float*)calloc(units, sizeof(float));
    for (int i = 0, j; i < units; i++) {
        for (j = 0; j < samples; j++) deriv->bias[i] += delta[j][i];
        deriv->bias[i] /= samples;
        for (j = 0; j < front_units; j++) deriv->weights[j][i] /= samples;
    }
    free_matrix(a_T, front_units);
    return deriv;
}
void print_weights(Weights* w, int decimal) {
	printf("Weights: [");
    int i, j;
	for (i = 0; i < w->row; i++) {
		if (i != 0) printf("\n%11s", "[");
        else printf("[");
		printf("%.*f", decimal, w->weights[i][0]);
		for (j = 1; j < w->col; j++) printf(", %.*f", decimal, w->weights[i][j]);
		printf("]");
	}
	printf("\n%11s%.*f", "[", decimal, w->bias[0]);
	for (j = 1; j < w->col; j++) printf(", %.*f", decimal, w->bias[j]);
    printf("]]\n");
}
void free_weights(Weights* w) {
    if (w->weights) {
        for (int i = 0; i < w->row; i++) free(w->weights[i]);
    }
    if (w->bias) free(w->bias);
    free(w);
}
void grad_descent(Weights* w, Weights* grad, float learning_rate, Weights* pre_velocity, float velocity_rate) {
	for (int i = 0, j; i < w->col; i++) {
		for (j = 0; j < w->row; j++) {
            grad->weights[j][i] *= learning_rate;
			grad->weights[j][i] += velocity_rate* pre_velocity->weights[j][i];
			w->weights[j][i] -= grad->weights[j][i];
			pre_velocity->weights[j][i] = grad->weights[j][i];
        }
        grad->bias[i] *= learning_rate;
        grad->bias[i] += velocity_rate* pre_velocity->bias[i];
        w->bias[i] -= grad->bias[i];
        pre_velocity->bias[i] = grad->bias[i];
	}
}
void adaptive_grad_descent(Weights* w, Weights* grad, float learning_rate, Weights* acc_grad, float epsilon) {
    for (int i = 0, j; i < w->col; i++) {
        for (j = 0; j < w->row; j++) {
            acc_grad->weights[j][i] += grad->weights[j][i]* grad->weights[j][i];
            w->weights[j][i] -= learning_rate* grad->weights[j][i] / sqrtf(acc_grad->weights[j][i] + epsilon);
        }
        acc_grad->bias[i] += grad->bias[i]* grad->bias[i];
        w->bias[i] -= learning_rate* grad->bias[i] / sqrtf(acc_grad->bias[i] + epsilon);
    }
}
void root_mean_square_propagation(Weights* w, Weights* grad, float learning_rate, Weights* acc_grad, float rho, 
                                    float epsilon, Weights* pre_velocity, float velocity_rate) {
    for (int i = 0, j; i < w->col; i++) {
        for (j = 0; j < w->row; j++) {
            acc_grad->weights[j][i] = rho* acc_grad->weights[j][i] + (1 - rho)* grad->weights[j][i]* grad->weights[j][i];
            pre_velocity->weights[j][i] = velocity_rate* pre_velocity->weights[j][i] + learning_rate* grad->weights[j][i] / sqrtf(acc_grad->weights[j][i] + epsilon);
            w->weights[j][i] -= pre_velocity->weights[j][i];
        }
        acc_grad->bias[i] = rho* acc_grad->bias[i]* + (1 - rho)* grad->bias[i]* grad->bias[i];
        pre_velocity->bias[i] = velocity_rate* pre_velocity->bias[i] + learning_rate* grad->bias[i] / sqrtf(acc_grad->bias[i] + epsilon);
        w->bias[i] -= pre_velocity->bias[i];
    }
}
void adaptive_delta_grad(Weights* w, Weights* grad, float learning_rate, Weights* acc_grad, float rho, 
                            float epsilon, Weights* delta) {
    for (int i = 0, j; i < w->col; i++) {
        for (j = 0; j < w->row; j++) {
            acc_grad->weights[j][i] = rho* acc_grad->weights[j][i] + (1 - rho)* grad->weights[j][i]* grad->weights[j][i];
            grad->weights[j][i] *= learning_rate* sqrtf((delta->weights[j][i] + epsilon) / (acc_grad->weights[j][i] + epsilon));
            w->weights[j][i] -= grad->weights[j][i];
            delta->weights[j][i] = rho* delta->weights[j][i] + (1 - rho)* grad->weights[j][i]* grad->weights[j][i];
        }
        acc_grad->bias[i] = rho* acc_grad->bias[i]* + (1 - rho)* grad->bias[i]* grad->bias[i];
        grad->bias[i] *= learning_rate* sqrtf((delta->bias[i] + epsilon) / (acc_grad->bias[i] + epsilon));
        w->bias[i] -= grad->bias[i];
        delta->bias[i] = rho* delta->bias[i] + (1 - rho)* grad->bias[i]* grad->bias[i];
    }
}
void adaptive_moment_estimation(Weights* w, Weights* grad, float learning_rate, Weights* acc_grad, float beta_1, 
                                float beta_2, float epsilon, Weights* pre_velocity, int times) {
    float beta_1t = powf(beta_1, (float) times), beta_2t = powf(beta_2, (float) times);
    float velo, accg;
    for (int i = 0, j; i < w->col; i++) {
        for (j = 0; j < w->row; j++) {
            pre_velocity->weights[j][i] = beta_1* pre_velocity->weights[j][i] + (1 - beta_1)* grad->weights[j][i];
            acc_grad->weights[j][i] = beta_2* acc_grad->weights[j][i] + (1 - beta_2)* grad->weights[j][i]* grad->weights[j][i];
            velo = pre_velocity->weights[j][i] / (1 - beta_1t), accg = acc_grad->weights[j][i] / (1 - beta_2t);
            w->weights[j][i] -= learning_rate* velo / (sqrtf(accg) + epsilon);
        }
        pre_velocity->bias[i] = beta_1* pre_velocity->bias[i] + (1 - beta_1)* grad->bias[i];
        acc_grad->bias[i] = beta_2* acc_grad->bias[i] + (1 - beta_2)* grad->bias[i]* grad->bias[i];
        velo = pre_velocity->bias[i] / (1 - beta_1t), accg = acc_grad->bias[i] / (1 - beta_2t);
        w->bias[i] -= learning_rate* velo / (sqrtf(accg) + epsilon);
    }
}
