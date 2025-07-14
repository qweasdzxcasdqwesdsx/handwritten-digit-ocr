using System.Diagnostics;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Windows;
using Microsoft.Extensions.Configuration;
using novaCount.Models;
using System.IO;


namespace novaCount.Services;

/// <summary>
/// 内核调用服务（改为 HTTP 调用 Flask API）
/// </summary>
public class CoreService : ICoreService
{
    private readonly IFileService  _fileService;
    private readonly CoreSettings  _settings;
    private readonly HttpClient    _http = new();   // 可换成 IHttpClientFactory 注入

    public CoreService(IFileService fileService)
    {
        _fileService = fileService;

        // 读取 CoreSettings 节点；找不到就抛异常
        _settings = App.Configuration.GetSection("CoreSettings").Get<CoreSettings>()
                    ?? throw new InvalidOperationException("未找到 CoreSettings 节点");
    }

    public ValueTask DisposeAsync()
    {
        _http.Dispose();
        GC.SuppressFinalize(this);
        return ValueTask.CompletedTask;
    }

    /// <summary>
    /// 把 TaskSettings 写成 novaCount_cfg.txt → 调用后端 API → 返回是否成功
    /// </summary>
    public async Task<bool> StartRecognition(TaskSettings settings)
    {
        // ① 补全模型路径
        settings.M2PathModel = Path.Combine(AppContext.BaseDirectory,
                                            _settings.ModelDir,
                                            "best.pt");

        // ② 写配置文件到内核目录下
        var cfgPath = Path.Combine(AppContext.BaseDirectory,
                                   _settings.Path,
                                   "novaCount_cfg.txt");

        if (await _fileService.WriteToFileAsync(settings.ToCoreString(), cfgPath) is not { } savedPath)
            return false;

        // ③ 组织 multipart/form-data
        using var form = new MultipartFormDataContent();
        await using var fs  = File.OpenRead(savedPath);
        form.Add(new StreamContent(fs), "config", Path.GetFileName(savedPath));

        try
        {
            // ④ POST 调用 Flask
            var resp = await _http.PostAsync(_settings.CoreApiUrl, form);

            // ⑤ 处理 HTTP 状态
            if (!resp.IsSuccessStatusCode)
            {
                var msg = await resp.Content.ReadAsStringAsync();
                MessageBox.Show($"内核 API 错误：{msg}", "后端错误",
                                 MessageBoxButton.OK, MessageBoxImage.Error);
                return false;
            }

            // ⑥ 解析返回 JSON
            var json = await resp.Content.ReadAsStringAsync();
            var doc  = JsonDocument.Parse(json);
            var status = doc.RootElement.GetProperty("status").GetString();

            if (status != "success")
            {
                var err = doc.RootElement.GetProperty("message").GetString();
                MessageBox.Show($"内核执行失败：{err}", "执行错误",
                                 MessageBoxButton.OK, MessageBoxImage.Error);
                return false;
            }

            return true;   // 成功
        }
        catch (HttpRequestException ex)
        {
            MessageBox.Show($"无法连接内核 API：{ex.Message}", "网络错误",
                             MessageBoxButton.OK, MessageBoxImage.Error);
            return false;
        }
        catch (Exception ex)
        {
            MessageBox.Show($"数据解析失败：{ex.Message}", "系统错误",
                             MessageBoxButton.OK, MessageBoxImage.Error);
            return false;
        }
    }
}
