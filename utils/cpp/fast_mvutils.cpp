
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include "fast_mvutils.h"

#define PI_2 2*M_PI

static inline double median(cv::Mat data) {
    auto begin = data.begin<int16_t>(),
         end = data.end<int16_t>();
    int size = data.total(),
        half_size = size / 2;
    // Sort half of the array
    std::partial_sort(begin, begin + half_size + 1, end);
    return (size % 2) ? (double)data.at<int16_t>(half_size)
        : (data.at<int16_t>(half_size-1) + data.at<int16_t>(half_size)) / 2.;
}

static inline cv::Mat vectorized_arctan2(cv::Mat &vectors) {
    cv::Mat result(1, vectors.total(), CV_64FC1);
    std::transform(
        vectors.begin<cv::Vec2s>(),
        vectors.end<cv::Vec2s>(),
        result.begin<double>(),
        [](const cv::Vec2s &v) -> double {
            return std::atan2(v(1), v(0));
        });
    return result;
}

static inline void find(cv::Mat &src, cv::Mat &mask, cv::Mat &out) {
    cv::Vec2s *p_src = src.ptr<cv::Vec2s>(),
              *p_out = out.ptr<cv::Vec2s>();
    uint8_t *p_mask = mask.ptr<uint8_t>();
    for (size_t i=0, j=0, n=mask.total(); i<n; i++) {
        if (p_mask[i]) {
            p_out[j] = p_src[i];
            j++;
        }
    }
}

static double fast_pvm_theta(cv::Mat vectors) {
    cv::Mat chan_vectors[2];
    cv::split(vectors, chan_vectors);
    cv::Mat mask_nonzero = (chan_vectors[0]!=0) | (chan_vectors[1]!=0);

    int n = cv::countNonZero(mask_nonzero);
    if (n == 0) {
        return 0.;
    } else {
        cv::Mat vectors_nonzero(1, n, vectors.type());
        find(vectors, mask_nonzero, vectors_nonzero);
        // Transform the input to polar coordinates in [-pi,+pi]
        cv::Mat theta_sort = vectorized_arctan2(vectors_nonzero);

        int num_nonoutliers, idx_median;
        if (n > 3) {
            // Define the number M of non-outlier vectors as n / 2
            num_nonoutliers = n/2;
            // sort the angles
            cv::sort(theta_sort, theta_sort,
                     cv::SORT_EVERY_ROW + cv::SORT_ASCENDING);
            // Calculate the difference in angles between adjacent vectors
            cv::Mat theta_sort_diff(theta_sort.size(), theta_sort.type());
            theta_sort_diff.colRange(0, n-1) =
                theta_sort.colRange(1, n) - theta_sort.colRange(0, n-1);
            // The difference of the last to the first is the angle that
            // complements the circle (2pi)
            theta_sort_diff.at<double>(n-1) = PI_2 -
                (theta_sort.at<double>(n-1) - theta_sort.at<double>(0));
            // Performs the cumulative sum of those differences (in place)
            double accum = 0.;
            for (auto it=theta_sort_diff.begin<double>(), end=theta_sort_diff.end<double>();
            it != end;
            ++it) {
                accum += *it;
                *it = accum;
            }
            // Sum up each adjacent sequence of M-1 differences of the angles
            int num_groups = theta_sort_diff.cols - (num_nonoutliers - 2);
            cv::Mat sum_adjacent_angles(1, num_groups, theta_sort_diff.type());

            sum_adjacent_angles.at<double>(0) = theta_sort_diff.at<double>(num_nonoutliers-2);
            sum_adjacent_angles.colRange(1, num_groups) =
                theta_sort_diff.colRange(num_nonoutliers-1, theta_sort_diff.cols)
                - theta_sort_diff.colRange(0, theta_sort_diff.cols - (num_nonoutliers-1));
            // The result index is the minimal sum-up
            cv::minMaxIdx(sum_adjacent_angles, NULL, NULL, &idx_median);
            // Calculate the median of the narrowest beam and of all vector norms
            idx_median += num_nonoutliers/2;
            if (idx_median == theta_sort.cols) // Rare case
                idx_median = 0;
        } else {
            num_nonoutliers = n;
            idx_median = n/2;
        }
        return (num_nonoutliers % 2) ? theta_sort.at<double>(idx_median)
            : (theta_sort.at<double>(idx_median-1) + theta_sort.at<double>(idx_median)) / 2.;
    }
}

double* c_fast_vectorized_pvm(int16_t (*vectors)[2], int shape[2]) {
    cv::Mat M_vectors(shape[0], shape[1], CV_16SC2, vectors);

    // Raise to power of two
    cv::Mat vectors_sqr;
    cv::pow(M_vectors, 2, vectors_sqr);

    // sum the channels to get squared norm of the vectors
    cv::Mat chan_vectors_sqr[2];
    cv::split(vectors_sqr, chan_vectors_sqr);
    chan_vectors_sqr[0] += chan_vectors_sqr[1];

    // allocate array filled with zeros for the result
    double (*result)[2] = (double(*)[2])calloc(shape[0], sizeof(double[2]));
    for (int i=0; i<shape[0]; i++) {
        double radius_sqr = median(chan_vectors_sqr[0].row(i));
        if (radius_sqr) {
            double radius = sqrt(radius_sqr),
                   theta = fast_pvm_theta(M_vectors.row(i));
            result[i][0] = radius * cos(theta);
            result[i][1] = radius * sin(theta);
        }
    }

    return (double*)result;
}
