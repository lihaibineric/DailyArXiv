import os
import time
import pytz
import shutil
import datetime
from typing import List, Dict
import urllib, urllib.request

import feedparser
from easydict import EasyDict

# 去除字符串中重复的空格，只保留一个空格
def remove_duplicated_spaces(text: str) -> str:
    return " ".join(text.split())

# 通过arXiv API请求论文数据，支持关键词、最大结果数和逻辑连接方式（OR/AND）
def request_paper_with_arXiv_api(keyword: str, max_results: int, link: str = "OR") -> List[Dict[str, str]]:
    # link参数只能为"OR"或"AND"
    assert link in ["OR", "AND"], "link should be 'OR' or 'AND'"
    # 用双引号包裹关键词，确保精确匹配
    keyword = "\"" + keyword + "\""
    # 构造arXiv API的查询URL，支持标题和摘要的联合搜索
    url = "http://export.arxiv.org/api/query?search_query=ti:{0}+{2}+abs:{0}&max_results={1}&sortBy=lastUpdatedDate".format(keyword, max_results, link)
    url = urllib.parse.quote(url, safe="%/:=&?~#+!$,;'@()*[]")
    # 发送请求并读取响应内容
    response = urllib.request.urlopen(url).read().decode('utf-8')
    # 使用feedparser解析返回的Atom格式数据
    feed = feedparser.parse(response)

    # 默认字段: Title, Authors, Abstract, Link, Tags, Comment, Date
    papers = []
    for entry in feed.entries:
        entry = EasyDict(entry)
        paper = EasyDict()

        # 处理标题，去除换行和多余空格
        paper.Title = remove_duplicated_spaces(entry.title.replace("\n", " "))
        # 处理摘要
        paper.Abstract = remove_duplicated_spaces(entry.summary.replace("\n", " "))
        # 处理作者列表
        paper.Authors = [remove_duplicated_spaces(_["name"].replace("\n", " ")) for _ in entry.authors]
        # 处理论文链接
        paper.Link = remove_duplicated_spaces(entry.link.replace("\n", " "))
        # 处理标签（领域分类）
        paper.Tags = [remove_duplicated_spaces(_["term"].replace("\n", " ")) for _ in entry.tags]
        # 处理评论信息
        paper.Comment = remove_duplicated_spaces(entry.get("arxiv_comment", "").replace("\n", " "))
        # 处理更新时间
        paper.Date = entry.updated

        papers.append(paper)
    return papers

# 过滤论文标签，只保留属于目标领域（如cs、stat）的论文
def filter_tags(papers: List[Dict[str, str]], target_fileds: List[str]=["cs", "stat"]) -> List[Dict[str, str]]:
    results = []
    for paper in papers:
        tags = paper.Tags
        for tag in tags:
            # 只保留主领域在目标列表中的论文
            if tag.split(".")[0] in target_fileds:
                results.append(paper)
                break
    return results

# 获取每日论文（带重试机制），避免API偶发性返回空列表
def get_daily_papers_by_keyword_with_retries(keyword: str, column_names: List[str], max_result: int, link: str = "OR", retries: int = 6) -> List[Dict[str, str]]:
    for _ in range(retries):
        papers = get_daily_papers_by_keyword(keyword, column_names, max_result, link)
        if len(papers) > 0: return papers
        else:
            print("Unexpected empty list, retrying...")
            time.sleep(60 * 30) # 等待30分钟后重试
    # 多次重试后仍失败，返回None
    return None

# 获取每日论文，按关键词和指定列筛选
def get_daily_papers_by_keyword(keyword: str, column_names: List[str], max_result: int, link: str = "OR") -> List[Dict[str, str]]:
    # 获取论文列表
    papers = request_paper_with_arXiv_api(keyword, max_result, link) # 默认字段: Title, Authors, Abstract, Link, Tags, Comment, Date
    # 只保留cs领域的论文（可扩展）
    papers = filter_tags(papers)
    # 为每个paper添加KiMi字段
    for paper in papers:
        paper["KiMi"] = "https://papers.cool/arxiv/" + paper["Link"].split("/")[-1]
    # 只保留需要展示的列
    papers = [{column_name: paper[column_name] for column_name in column_names} for paper in papers]
    return papers

# 生成Markdown表格，支持摘要、标签等字段的折叠显示
def generate_table(papers: List[Dict[str, str]], ignore_keys: List[str] = []) -> str:
    formatted_papers = []
    keys = papers[0].keys()
    for paper in papers:
        # 处理固定列
        formatted_paper = EasyDict()
        # 标题和链接，使用Markdown超链接格式
        formatted_paper.Title = "**" + "[{0}]({1})".format(paper["Title"], paper["Link"]) + "**"
        # 日期格式化（只保留年月日）
        formatted_paper.Date = paper["Date"].split("T")[0]
        # 处理其他列
        for key in keys:
            if key in ["Title", "Link", "Date"] or key in ignore_keys:
                continue
            elif key == "Abstract":
                # 摘要添加可折叠显示
                formatted_paper[key] = "<details><summary>Show</summary><p>{0}</p></details>".format(paper[key])
            elif key == "Authors":
                # 只显示第一作者，后面加et al.
                formatted_paper[key] = paper[key][0] + " et al."
            elif key == "Tags":
                tags = ", ".join(paper[key])
                # 标签较多时折叠显示
                if len(tags) > 10:
                    formatted_paper[key] = "<details><summary>{0}...</summary><p>{1}</p></details>".format(tags[:5], tags)
                else:
                    formatted_paper[key] = tags
            elif key == "Comment":
                if paper[key] == "":
                    formatted_paper[key] = ""
                elif len(paper[key]) > 20:
                    formatted_paper[key] = "<details><summary>{0}...</summary><p>{1}</p></details>".format(paper[key][:5], paper[key])
                else:
                    formatted_paper[key] = paper[key]
            elif key == "KiMi":
                formatted_paper[key] = "https://papers.cool/arxiv/" + paper["Link"].split('/')[-1]
        formatted_papers.append(formatted_paper)

    # 生成表头
    columns = formatted_papers[0].keys()
    # 表头加粗
    columns = ["**" + column + "**" for column in columns]
    header = "| " + " | ".join(columns) + " |"
    header = header + "\n" + "| " + " | ".join(["---"] * len(formatted_papers[0].keys())) + " |"
    # 生成表体
    body = ""
    for paper in formatted_papers:
        body += "\n| " + " | ".join(paper.values()) + " |"
    return header + body

# 备份README.md和ISSUE_TEMPLATE.md文件，防止写入失败导致数据丢失
def back_up_files():
    shutil.move("README.md", "README.md.bk")
    shutil.move(".github/ISSUE_TEMPLATE.md", ".github/ISSUE_TEMPLATE.md.bk")

# 恢复README.md和ISSUE_TEMPLATE.md文件（从备份还原）
def restore_files():
    shutil.move("README.md.bk", "README.md")
    shutil.move(".github/ISSUE_TEMPLATE.md.bk", ".github/ISSUE_TEMPLATE.md")

# 删除备份文件，清理临时文件
def remove_backups():
    os.remove("README.md.bk")
    os.remove(".github/ISSUE_TEMPLATE.md.bk")

# 获取北京时间，格式如"March 1, 2021"
def get_daily_date():
    beijing_timezone = pytz.timezone('Asia/Shanghai')
    today = datetime.datetime.now(beijing_timezone)
    return today.strftime("%B %d, %Y")
