/**
 * Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * @brief 他のライブラリ等に依存しない基本的なマクロ群
 */
#pragma once

// 名前空間の開始・終了
#define ASRC_NAMESPACE_BEGIN(name) namespace name {
#define ASRC_INLINE_NAMESPACE_BEGIN(name) inline namespace name {
#define ASRC_NAMESPACE_END(name) }

/// @brief コンマ含みの文字列をそのまま文字列リテラル化するマクロ
#define ASRC_INTERNAL_TOSTRING_VA_ARGS(...) #__VA_ARGS__

/// @brief 標準出力へのデバッグ用マクロ
#define DEBUG_PRINT_EXPRESSION(ex) std::cout<<#ex<<"="<<ex<<std::endl;

// ループマクロ用デリミタ
#define ASRC_INTERNAL_COMMA(dummy) ,
#define ASRC_INTERNAL_SEMICOLON(dummy) ;

// ループマクロ
// func(x)をdelimiter(x)で区切って並べる
#define ASRC_INTERNAL_FOREACH_0(func, delimiter)
#define ASRC_INTERNAL_FOREACH_1(func, delimiter, x) func(x)
#define ASRC_INTERNAL_FOREACH_2(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_1(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_3(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_2(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_4(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_3(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_5(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_4(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_6(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_5(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_7(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_6(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_8(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_7(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_9(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_8(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_10(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_9(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_11(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_10(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_12(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_11(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_13(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_12(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_14(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_13(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_15(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_14(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_16(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_15(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_17(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_16(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_18(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_17(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_19(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_18(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_20(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_19(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_21(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_20(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_22(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_21(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_23(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_22(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_24(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_23(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_25(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_24(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_26(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_25(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_27(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_26(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_28(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_27(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_29(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_28(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_30(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_29(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_31(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_30(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_32(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_31(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_33(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_32(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_34(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_33(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_35(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_34(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_36(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_35(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_37(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_36(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_38(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_37(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_39(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_38(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_40(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_39(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_41(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_40(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_42(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_41(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_43(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_42(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_44(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_43(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_45(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_44(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_46(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_45(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_47(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_46(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_48(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_47(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_49(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_48(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_50(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_49(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_51(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_50(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_52(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_51(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_53(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_52(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_54(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_53(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_55(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_54(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_56(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_55(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_57(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_56(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_58(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_57(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_59(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_58(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_60(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_59(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_61(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_60(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_62(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_61(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_63(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_62(func, delimiter, __VA_ARGS__)
#define ASRC_INTERNAL_FOREACH_64(func, delimiter, x, ...) func(x) delimiter(x) ASRC_INTERNAL_FOREACH_63(func, delimiter, __VA_ARGS__)
#ifndef __DOXYGEN__
    #define ASRC_INTERNAL_GET_MACRO(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, _62, _63, _64, NAME, ...) NAME
#else
    // Doxygenは__VA_OPT__に対応していないため、__VA_ARGS__が0個にならないようにマクロを構成しないといけない
    #define ASRC_INTERNAL_GET_MACRO(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, _62, _63, _64, NAME, ...) NAME
#endif /*!__DOXYGEN__*/

#define ASRC_INTERNAL_EXPAND( x ) x

#ifndef __DOXYGEN__
/// @brief ループマクロ。func(x)をdelimiter(x)で区切って並べる
#define ASRC_INTERNAL_FOREACH(func, delimiter, ...) \
    ASRC_INTERNAL_EXPAND(ASRC_INTERNAL_GET_MACRO(_0 __VA_OPT__(,) __VA_ARGS__, \
        ASRC_INTERNAL_FOREACH_64, \
        ASRC_INTERNAL_FOREACH_63, \
        ASRC_INTERNAL_FOREACH_62, \
        ASRC_INTERNAL_FOREACH_61, \
        ASRC_INTERNAL_FOREACH_60, \
        ASRC_INTERNAL_FOREACH_59, \
        ASRC_INTERNAL_FOREACH_58, \
        ASRC_INTERNAL_FOREACH_57, \
        ASRC_INTERNAL_FOREACH_56, \
        ASRC_INTERNAL_FOREACH_55, \
        ASRC_INTERNAL_FOREACH_54, \
        ASRC_INTERNAL_FOREACH_53, \
        ASRC_INTERNAL_FOREACH_52, \
        ASRC_INTERNAL_FOREACH_51, \
        ASRC_INTERNAL_FOREACH_50, \
        ASRC_INTERNAL_FOREACH_49, \
        ASRC_INTERNAL_FOREACH_48, \
        ASRC_INTERNAL_FOREACH_47, \
        ASRC_INTERNAL_FOREACH_46, \
        ASRC_INTERNAL_FOREACH_45, \
        ASRC_INTERNAL_FOREACH_44, \
        ASRC_INTERNAL_FOREACH_43, \
        ASRC_INTERNAL_FOREACH_42, \
        ASRC_INTERNAL_FOREACH_41, \
        ASRC_INTERNAL_FOREACH_40, \
        ASRC_INTERNAL_FOREACH_39, \
        ASRC_INTERNAL_FOREACH_38, \
        ASRC_INTERNAL_FOREACH_37, \
        ASRC_INTERNAL_FOREACH_36, \
        ASRC_INTERNAL_FOREACH_35, \
        ASRC_INTERNAL_FOREACH_34, \
        ASRC_INTERNAL_FOREACH_33, \
        ASRC_INTERNAL_FOREACH_32, \
        ASRC_INTERNAL_FOREACH_31, \
        ASRC_INTERNAL_FOREACH_30, \
        ASRC_INTERNAL_FOREACH_29, \
        ASRC_INTERNAL_FOREACH_28, \
        ASRC_INTERNAL_FOREACH_27, \
        ASRC_INTERNAL_FOREACH_26, \
        ASRC_INTERNAL_FOREACH_25, \
        ASRC_INTERNAL_FOREACH_24, \
        ASRC_INTERNAL_FOREACH_23, \
        ASRC_INTERNAL_FOREACH_22, \
        ASRC_INTERNAL_FOREACH_21, \
        ASRC_INTERNAL_FOREACH_20, \
        ASRC_INTERNAL_FOREACH_19, \
        ASRC_INTERNAL_FOREACH_18, \
        ASRC_INTERNAL_FOREACH_17, \
        ASRC_INTERNAL_FOREACH_16, \
        ASRC_INTERNAL_FOREACH_15, \
        ASRC_INTERNAL_FOREACH_14, \
        ASRC_INTERNAL_FOREACH_13, \
        ASRC_INTERNAL_FOREACH_12, \
        ASRC_INTERNAL_FOREACH_11, \
        ASRC_INTERNAL_FOREACH_10, \
        ASRC_INTERNAL_FOREACH_9, \
        ASRC_INTERNAL_FOREACH_8, \
        ASRC_INTERNAL_FOREACH_7, \
        ASRC_INTERNAL_FOREACH_6, \
        ASRC_INTERNAL_FOREACH_5, \
        ASRC_INTERNAL_FOREACH_4, \
        ASRC_INTERNAL_FOREACH_3, \
        ASRC_INTERNAL_FOREACH_2, \
        ASRC_INTERNAL_FOREACH_1, \
        ASRC_INTERNAL_FOREACH_0)(func, delimiter __VA_OPT__(,) __VA_ARGS__))
#else
/// @brief ループマクロ。func(x)をdelimiter(x)で区切って並べる
#define ASRC_INTERNAL_FOREACH(func, delimiter, ...) \
    ASRC_INTERNAL_EXPAND(ASRC_INTERNAL_GET_MACRO(__VA_ARGS__, \
        ASRC_INTERNAL_FOREACH_64, \
        ASRC_INTERNAL_FOREACH_63, \
        ASRC_INTERNAL_FOREACH_62, \
        ASRC_INTERNAL_FOREACH_61, \
        ASRC_INTERNAL_FOREACH_60, \
        ASRC_INTERNAL_FOREACH_59, \
        ASRC_INTERNAL_FOREACH_58, \
        ASRC_INTERNAL_FOREACH_57, \
        ASRC_INTERNAL_FOREACH_56, \
        ASRC_INTERNAL_FOREACH_55, \
        ASRC_INTERNAL_FOREACH_54, \
        ASRC_INTERNAL_FOREACH_53, \
        ASRC_INTERNAL_FOREACH_52, \
        ASRC_INTERNAL_FOREACH_51, \
        ASRC_INTERNAL_FOREACH_50, \
        ASRC_INTERNAL_FOREACH_49, \
        ASRC_INTERNAL_FOREACH_48, \
        ASRC_INTERNAL_FOREACH_47, \
        ASRC_INTERNAL_FOREACH_46, \
        ASRC_INTERNAL_FOREACH_45, \
        ASRC_INTERNAL_FOREACH_44, \
        ASRC_INTERNAL_FOREACH_43, \
        ASRC_INTERNAL_FOREACH_42, \
        ASRC_INTERNAL_FOREACH_41, \
        ASRC_INTERNAL_FOREACH_40, \
        ASRC_INTERNAL_FOREACH_39, \
        ASRC_INTERNAL_FOREACH_38, \
        ASRC_INTERNAL_FOREACH_37, \
        ASRC_INTERNAL_FOREACH_36, \
        ASRC_INTERNAL_FOREACH_35, \
        ASRC_INTERNAL_FOREACH_34, \
        ASRC_INTERNAL_FOREACH_33, \
        ASRC_INTERNAL_FOREACH_32, \
        ASRC_INTERNAL_FOREACH_31, \
        ASRC_INTERNAL_FOREACH_30, \
        ASRC_INTERNAL_FOREACH_29, \
        ASRC_INTERNAL_FOREACH_28, \
        ASRC_INTERNAL_FOREACH_27, \
        ASRC_INTERNAL_FOREACH_26, \
        ASRC_INTERNAL_FOREACH_25, \
        ASRC_INTERNAL_FOREACH_24, \
        ASRC_INTERNAL_FOREACH_23, \
        ASRC_INTERNAL_FOREACH_22, \
        ASRC_INTERNAL_FOREACH_21, \
        ASRC_INTERNAL_FOREACH_20, \
        ASRC_INTERNAL_FOREACH_19, \
        ASRC_INTERNAL_FOREACH_18, \
        ASRC_INTERNAL_FOREACH_17, \
        ASRC_INTERNAL_FOREACH_16, \
        ASRC_INTERNAL_FOREACH_15, \
        ASRC_INTERNAL_FOREACH_14, \
        ASRC_INTERNAL_FOREACH_13, \
        ASRC_INTERNAL_FOREACH_12, \
        ASRC_INTERNAL_FOREACH_11, \
        ASRC_INTERNAL_FOREACH_10, \
        ASRC_INTERNAL_FOREACH_9, \
        ASRC_INTERNAL_FOREACH_8, \
        ASRC_INTERNAL_FOREACH_7, \
        ASRC_INTERNAL_FOREACH_6, \
        ASRC_INTERNAL_FOREACH_5, \
        ASRC_INTERNAL_FOREACH_4, \
        ASRC_INTERNAL_FOREACH_3, \
        ASRC_INTERNAL_FOREACH_2, \
        ASRC_INTERNAL_FOREACH_1)(func, delimiter, __VA_ARGS__))
#endif/*!__DOXYGEN__*/

//
// Doxygen使用時用のマクロ
// 幾つかの必須引数と0個以上の可変引数をとるマクロを、__VA_OPT__を使わずに実現する。
//
#ifdef __DOXYGEN__
/// @brief [For internal use] __VA_ARGSの有無によって呼び出すマクロを切り替えるマクロ。これは必須引数が1つの場合。
#define ASRC_INTERNAL_1_AND_VA_ARGS(no_va_arg,with_va_args,...)\
    ASRC_INTERNAL_EXPAND(ASRC_INTERNAL_GET_MACRO(__VA_ARGS__, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        no_va_arg)(__VA_ARGS__))

/// @brief [For internal use] __VA_ARGSの有無によって呼び出すマクロを切り替えるマクロ。これは必須引数が2つの場合。
#define ASRC_INTERNAL_2_AND_VA_ARGS(no_va_arg,with_va_args,...)\
    ASRC_INTERNAL_EXPAND(ASRC_INTERNAL_GET_MACRO(__VA_ARGS__, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        no_va_arg)(__VA_ARGS__))

/// @brief [For internal use] __VA_ARGSの有無によって呼び出すマクロを切り替えるマクロ。これは必須引数が3つの場合。
#define ASRC_INTERNAL_3_AND_VA_ARGS(no_va_arg,with_va_args,...)\
    ASRC_INTERNAL_EXPAND(ASRC_INTERNAL_GET_MACRO(__VA_ARGS__, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        with_va_args, \
        no_va_arg)(__VA_ARGS__))

/// @brief [For internal use] __VA_ARGSの有無によって呼び出すマクロを切り替えるマクロ。Doxygen用
#define ASRC_INTERNAL_DEFINE_VA_MACRO(macro_name,num_mandatory_arg,...) \
    ASRC_INTERNAL_EXPAND(ASRC_INTERNAL_##num_mandatory_arg##_AND_VA_ARGS( \
        macro_name##_IMPL_NO_VA_ARG, \
        macro_name##_IMPL_VA_ARGS, \
        __VA_ARGS__))

#endif/*__DOXYGEN__*/
