namespace novaCount.Models
{
    /// <summary>
    /// 内核与模型的配置参数（由 appsettings.json 注入）
    /// </summary>
    public class CoreSettings
    {
        /// <summary>
        /// 内核 exe / 资源所在文件夹
        /// </summary>
        public required string Path { get; init; }

        /// <summary>
        /// 识别模型文件夹
        /// </summary>
        public required string ModelDir { get; init; }

        /// <summary>
        /// Flask 后端 API 地址，例如 "http://127.0.0.1:5000/api/run"
        /// </summary>
        public required string CoreApiUrl { get; init; }
    }
}