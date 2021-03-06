#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "stex.h"
#include "util.h"
#include "matrix.h"

// Remaps 'labels' with 'size' elements according to
// 'factors', an array with the label mapping.
// Params:
void remap(int *labels, size_t size, int *factors) {
    size_t i;
    for(i = 0; i < size; ++i) {
        labels[i] = factors[labels[i]];
    }
}

// Defuzzifies fuzzy matrix 'fuzmtx' by using the first
// maxima method.
// Params:
//  fuzmtx - an object x class fuzzy matrix.
// Return:
//  An array of size 'fuzmtx->nrow' with the crip values for each
//  object.
int* defuz(st_matrix *fuzmtx) {
    size_t i;
    size_t j;
    double maxval;
    double val;
    int *labels = malloc(sizeof(int) * fuzmtx->nrow);
    for(i = 0; i < fuzmtx->nrow; ++i) {
        maxval = get(fuzmtx, i, 0);
        labels[i] = 0;
        for(j = 1; j < fuzmtx->ncol; ++j) {
            val = get(fuzmtx, i, j);
            if(val > maxval) {
                maxval = val;
                labels[i] = j;
            }
        }
    }
    return labels;
}

// Creates a confusion matrix.
// Params:
//  labels - the labels of each object.
//  pred - the predicted labels of each object.
//  size - the size of the 'labels' and 'pred' arrays.
// Return:
//  A confusion matrix size x size.
st_matrix* confusion(int *labels, int *pred, size_t size) {
    size_t classc = max(labels, size) + 1;
    st_matrix *confmtx = malloc(sizeof(st_matrix));
    init_st_matrix(confmtx, classc, classc);
    setall(confmtx, 0.0);
    size_t i;
    double val;
    for(i = 0; i < size; ++i) {
        val = get(confmtx, labels[i], pred[i]) + 1.0;
        set(confmtx, labels[i], pred[i], val);
        set(confmtx, pred[i], labels[i], val);

    }
    return confmtx;
}

// Computes the fuzzy partition coefficient.
// Params:
//  fuzmtx - an 'object x class' fuzzy matrix.
// Return:
//  The partition coefficient validation index.
double partcoef(st_matrix *fuzmtx) {
    size_t i;
    size_t k;
    double ret = 0.0;
    for(k = 0; k < fuzmtx->ncol; ++k) {
        for(i = 0; i < fuzmtx->nrow; ++i) {
            ret += pow(get(fuzmtx, i, k), 2.0);
        }
    }
    return ret / ((double) fuzmtx->nrow);
}

// Computes the fuzzy modified partition coefficient.
// Params:
//  fuzmtx - an 'object x class' fuzzy matrix.
// Return:
//  The partition modified coefficient validation index.
double modpcoef(st_matrix *fuzmtx) {
    double pc = partcoef(fuzmtx);
    double classc = fuzmtx->ncol;
    return 1.0 - (classc / (classc - 1.0)) * (1.0 - pc);
}

// Computes the partition entropy of a fuzzy membership matrix.
// Params:
//  fuzmtx - an 'object x class' fuzzy matrix.
// Return:
//  The partition entropy validation index.
double partent(st_matrix *fuzmtx) {
    size_t i;
    size_t k;
    double ret = 0.0;
    double val;
    for(k = 0; k < fuzmtx->ncol; ++k) {
        for(i = 0; i < fuzmtx->nrow; ++i) {
            val = get(fuzmtx, i, k);
            if(val) {
                ret +=  val * log(val);
            }
        }
    }
    return -(ret / ((double) fuzmtx->nrow));
}

// Computes the corrected Rand index.
// Params:
//  labels - the labels of each object.
//  pred - the predicted labels of each object.
//  size - the size of the 'labels' and 'pred' arrays.
// Return:
//  The corrected Rand index.
double corand(int *labels, int *pred, size_t size) {
    size_t i;
    size_t j;
    size_t last = size - 1;
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    double d = 0.0;
    double val_labels;
    double val_pred;
    double val_labels_neg;
    double val_pred_neg;
    for(i = 0; i < last; ++i) {
        for(j = i + 1; j < size; ++j) {
            val_labels = labels[i] == labels[j];
            val_pred = pred[i] == pred[j];
            val_labels_neg = 1.0 - val_labels;
            val_pred_neg = 1.0 - val_pred;
            a += val_labels * val_pred;
            b += val_labels_neg * val_pred;
            c += val_labels * val_pred_neg;
            d += val_labels_neg * val_pred_neg;
        }
    }
    double p = a + b + c + d;
    double term = ((a + b) * (a + c) + (c + d) * (b + d)) * (1.0 / p);
    return ((a + d) - term) / (p - term);
}

double avg_intra_dist(st_matrix *fuzmtx, st_matrix *dist, double mfuz) {
    size_t clustc = fuzmtx->ncol;
    size_t objc = fuzmtx->nrow;
    size_t i;
    size_t k;
    double val;
    double sumnum;
    double sumden;
    double ret = 0.0;
    for(k = 0; k < clustc; ++k) {
        sumnum = 0.0;
        sumden = 0.0;
        for(i = 0; i < objc; ++i) {
            val = pow(get(fuzmtx, i, k), mfuz);
            sumden += val;
            sumnum += val * get(dist, i, k);
        }
        ret += (sumnum / sumden);
    }
    return (ret / (clustc * objc));
}

silhouet* crispsil(st_matrix *groups, st_matrix *dmatrix) {
    size_t i;
    size_t objc = 0;
    silhouet *sil = malloc(sizeof(silhouet));
    for(i = 0; i < groups->nrow; ++i) {
        objc += get(groups, i, 0);
    }
    sil->objc = objc;
    sil->clustc = groups->nrow;
    sil->objsil = malloc(objc * sizeof(double));
    sil->clustsil = calloc(groups->nrow, sizeof(double));
    sil->avgsil = 0.0;
    size_t c;
    size_t h;
    size_t k;
    size_t size;
    size_t othersize;
    double intra;
    size_t obj;
    double intermin;
    double intercur;
    double val;
    bool first;
    for(k = 0; k < groups->nrow; ++k) {
        size = get(groups, k, 0);
        for(i = 1; i <= size; ++i) {
            intra = 0.0;
            obj = get(groups, k, i);
            for(h = 1; h <= size; ++h) {
                intra += get(dmatrix, obj, get(groups, k, h));
            }
            intra /= size;
            first = true;
            for(c = 0; c < groups->nrow; ++c) {
                if(c != k) {
                    othersize = get(groups, c, 0);
                    intercur = 0.0;
                    for(h = 1; h <= othersize; ++h) {
                        intercur += get(dmatrix, obj, get(groups, c, h));
                    }
                    intercur /= othersize;
                    if(first || intercur < intermin) {
                        intermin = intercur;
                        first = false;
                    }
                }
            }
            val = (intermin - intra) /
                (intermin > intra ? intermin : intra);
            sil->objsil[obj] = val;
            sil->clustsil[k] += val;
            sil->avgsil += val;
        }
        sil->clustsil[k] /= size;
    }
    sil->avgsil /= objc;
    return sil;
}

silhouet* simplesil(int *pred, st_matrix *cent_dist) {
    silhouet *ssil = malloc(sizeof(silhouet));
    ssil->objc = cent_dist->nrow;
    ssil->clustc = cent_dist->ncol;
    ssil->objsil = malloc(ssil->objc * sizeof(double));
    ssil->clustsil = calloc(ssil->clustc, sizeof(double));
    ssil->avgsil = 0.0;
    int i;
    int j;
    double intra;
    double inter;
    double val;
    bool first;
    size_t clust_card[ssil->clustc];
    for(i = 0; i < ssil->clustc; ++i) {
        clust_card[i] = 0.0;
    }
    int predclust;
    for(i = 0; i < cent_dist->nrow; ++i) {
        first = true;
        predclust = pred[i];
        for(j = 0; j < cent_dist->ncol; ++j) {
            val = get(cent_dist, i, j);
            if(j != predclust) {
                if(first || val < inter) {
                    inter = val;
                    first = false;
                }
            } else {
                intra = val;
            }
        }
        val = (inter - intra) / (inter > intra ? inter : intra);
        ssil->objsil[i] = val;
        ssil->clustsil[predclust] += val;
        ssil->avgsil += val;
        clust_card[predclust]++;
    }
    for(i = 0; i < ssil->clustc; ++i) {
        ssil->clustsil[i] /= clust_card[i];
    }
    ssil->avgsil /= ssil->objc;
    return ssil;
}

silhouet* fuzzysil(silhouet *sil, st_matrix *groups, st_matrix *memb,
                    double alpha) {
    silhouet *fsil = malloc(sizeof(silhouet));
    fsil->objc = sil->objc;
    fsil->clustc = sil->clustc;
    fsil->objsil = malloc(fsil->objc * sizeof(double));
    fsil->clustsil = calloc(fsil->clustc, sizeof(double));
    fsil->avgsil = 0.0;
    size_t i;
    size_t k;
    size_t c;
    size_t size;
    size_t obj;
    double fst_memb;
    double snd_memb;
    double cur_memb;
    double val;
    double sumden_clust;
    double sumden_all = 0.0;
    for(k = 0; k < fsil->clustc; ++k) {
        size = get(groups, k, 0);
        sumden_clust = 0.0;
        for(i = 1; i <= size; ++i) {
            obj = get(groups, k, i);
            fst_memb = get(memb, obj, 0);
            snd_memb = get(memb, obj, 1);
            if(snd_memb > fst_memb) {
                cur_memb = snd_memb;
                snd_memb = fst_memb;
                fst_memb = cur_memb;
            }
            for(c = 2; c < memb->ncol; ++c) {
                cur_memb = get(memb, obj, c);
                if(cur_memb > fst_memb) {
                    snd_memb = fst_memb;
                    fst_memb = cur_memb;
                } else if(cur_memb > snd_memb) {
                    snd_memb = cur_memb;
                }
            }
            fsil->objsil[obj] = sil->objsil[obj];
            val = pow(fst_memb - snd_memb, alpha);
            fsil->clustsil[k] += val * sil->objsil[obj];
            sumden_clust += val;
        }
        fsil->avgsil += fsil->clustsil[k];
        fsil->clustsil[k] /= sumden_clust;
        sumden_all += sumden_clust;
    }
    fsil->avgsil /= sumden_all;
    return fsil;
}

void print_groups(st_matrix *groups) {
    size_t i;
    size_t j;
    size_t size;
    for(i = 0; i < groups->nrow; ++i) {
        size = get(groups, i, 0);
        printf("Group %u (%u members):\n", i, size);
        for(j = 1; j <= size; ++j) {
            printf("%.0f ", get(groups, i, j));
        }
        printf("\n");
    }
}

//st_matrix* asgroups(int *labels, size_t size, size_t card) {
//    st_matrix *groups = malloc(sizeof(st_matrix));
//    init_st_matrix(groups, card, size + 1);
//    int i;
//    for(i = 0; i < card; ++i) {
//        groups->mtx[i * groups->ncol] = 0.0;
//    }
//    printf("Init:\n");
//    print_groups(groups);
//    int label;
//    int elemc;
//    for(i = 0; i < size; ++i) {
//        label = labels[i];
//        printf("Obj %u, label %u\n", i, label);
//        elemc = groups->mtx[label * groups->ncol];
//        groups->mtx[label * groups->ncol + elemc] = i;
//        groups->mtx[label * groups->ncol] = elemc + 1;
//        print_groups(groups);
//    }
//    return groups;
//}

st_matrix* asgroups(int *labels, size_t size, size_t card) {
    st_matrix *groups = malloc(sizeof(st_matrix));
    init_st_matrix(groups, card, size + 1);
    int i;
    for(i = 0; i < card; ++i) {
        set(groups, i, 0, 0.0);
    }
//    printf("Init:\n");
//    print_groups(groups);
    int label;
    int elemc;
    for(i = 0; i < size; ++i) {
        label = labels[i];
//        printf("Obj %u, label %u\n", i, label);
        elemc = get(groups, label, 0) + 1;
        set(groups, label, elemc, i);
        set(groups, label, 0, elemc);
//        print_groups(groups);
    }
    return groups;
}

// Prints the objects by group.
// Params:
//  labels - the labels of each object.
//  size - the size of the 'labels' and 'pred' arrays.
//  card - the number of different groups in 'labels'.
//void print_groups(int *labels, size_t size, size_t card) {
//    size_t groups[card][size - 1];
//    size_t groupsc[card];
//    size_t i;
//    for(i = 0; i < card; ++i) {
//        groupsc[i] = 0;
//    }
//    int label;
//    for(i = 0; i < size; ++i) {
//        label = labels[i];
//        groups[label][groupsc[label]] = i;
//        groupsc[label] = groupsc[label] + 1;
//    }
//    size_t j;
//    for(i = 0; i < card; ++i) {
//        printf("Group %u (%u members):\n", i, groupsc[i]);
//        for(j = 0; j < groupsc[i]; ++j) {
//            printf("%u ", groups[i][j]);
//        }
//        printf("\n");
//    }
//}

silhouet* avg_silhouet(silhouet *s1, silhouet *s2) {
    if(s1->objc != s2->objc || s1->clustc != s2->clustc) {
        return NULL;
    }
    size_t i;
    for(i = 0; i < s1->objc; ++i) {
        s1->objsil[i] = (s1->objsil[i] + s2->objsil[i]) / 2.0;
    }
    for(i = 0; i < s1->clustc; ++i) {
        s1->clustsil[i] = (s1->clustsil[i] + s2->clustsil[i]) / 2.0;
    }
    s1->avgsil = (s1->avgsil + s2->avgsil);
    return s1;
}

void free_silhouet(silhouet *s) {
    free(s->objsil);
    free(s->clustsil);
}

void print_silhouet(silhouet *s) {
    printf("Object silhouette:\n");
    size_t i;
    size_t last = s->objc - 1;
    for(i = 0; i < last; ++i) {
        printf("(%d, %lf) ", i, s->objsil[i]);
    }
    printf("(%d, %lf)\n", i, s->objsil[i]);
    printf("Average silhouette by cluster:\n");
    last = s->clustc - 1;
    for(i = 0; i < last; ++i) {
        printf("(%d, %lf) ", i, s->clustsil[i]);
    }
    printf("(%d, %lf)\n", i, s->clustsil[i]);
    printf("Average silhouette: %lf\n", s->avgsil);
}
