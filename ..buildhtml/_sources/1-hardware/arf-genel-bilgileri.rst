.. _arf-genel-bilgileri:

===================================
ARF Kümesi Hakkında Genel Bilgiler
===================================


ARF hesaplama kümesi için ``arf-ui1`` ve ``arf-ui2`` olmak üzere iki tane sunucu, kullanıcı arayüzü olarak hizmete sunulmustur. Bu hesaplama kümesinde bulunan ``orfoz`` ismini atadığımız hesaplama sunucularında iş koşturmak için arf kullanıcı arayüzleri kullanılacaktır. 

Kümenin genel özellikleri aşağıdaki gibidir:

- Hesaplama sunucu sayısı: 504
- Hesaplama sunucu adı: orfoz[1-504]
- İşlemci:  2x Intel(R) Xeon(R) Platinum 8480+ (toplam 112 çekirdek)
- Bellek:   256 GB
- Network: 200Gbit NDR infiniband
- Tmp : 750GB name
- Merkezi depolama : `/arf` (1.6Pbyte)
- Ev dizini: `/arf/home`
- Yazılım: `/arf/sw/(apps,comp,lib)`
- İşletim sistemi: Rocky Linux 9.2

Güncel kurulu yazılımları 

.. code-block:: bash

	module available

komutu ile listeleyebilirsiniz. Örnek SLURM betik dosyaları ``/arf/sw/scripts`` dizininde bulunmaktadır.
