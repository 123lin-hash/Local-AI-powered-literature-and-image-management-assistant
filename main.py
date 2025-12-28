import argparse
import os

# 导入你的两个系统
from paper_manager import add_paper, batch_sort, search_paper
from image_manager import add_images, search_image


def main():
    parser = argparse.ArgumentParser(
        description="智能论文管理 & 以文搜图系统"
    )

    subparsers = parser.add_subparsers(dest="command")

    # 添加 / 分类论文
    add_paper_parser = subparsers.add_parser(
        "add_paper",
        help="添加并自动分类论文（PDF 文件或目录）"
    )
    add_paper_parser.add_argument(
        "path",
        type=str,
        help="PDF 文件路径 或 包含 PDF 的目录"
    )

    # 搜索论文
    search_paper_parser = subparsers.add_parser(
        "search_paper",
        help="语义搜索论文"
    )
    search_paper_parser.add_argument(
        "query",
        type=str,
        help="自然语言查询，如：Transformer 的核心架构是什么？"
    )
    search_paper_parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="返回的论文数量（默认 1）"
    )

    # 图片入库
    add_images_parser = subparsers.add_parser(
        "add_images",
        help="批量入库图片"
    )
    add_images_parser.add_argument(
        "dir",
        type=str,
        help="图片目录路径"
    )

    # 以文搜图
    search_image_parser = subparsers.add_parser(
        "search_image",
        help="通过文本搜索图片"
    )
    search_image_parser.add_argument(
        "query",
        type=str,
        help="文本描述，如：sunset at the beach"
    )
    search_image_parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="返回图片数量（默认 1）"
    )

    args = parser.parse_args()

    # 命令分发
    if args.command == "add_paper":
        if os.path.isdir(args.path):
            batch_sort(args.path)
        else:
            add_paper(args.path)

    elif args.command == "search_paper":
        results = search_paper(args.query, top_k=args.top_k)
        print("\n最相关论文：")
        for r in results:
            print("-", r)

    elif args.command == "add_images":
        add_images(args.dir)

    elif args.command == "search_image":
        results = search_image(args.query, top_k=args.top_k)
        print("\n最匹配图片：")
        for r in results:
            print("-", r)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
